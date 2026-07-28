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

Environment variables:
- SEQUENTIAL_ALL=1: Force sequential routing (cost update after every net)
- USE_GPU=1: Enable GPU acceleration
- INCREMENTAL_COST_UPDATE=1: Only update costs for edges that changed
- ORTHO_CPU_ONLY=1: Force CPU-only mode (no GPU)
"""

# Standard library
import copy
import logging
import time
import random
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
from collections import defaultdict

# Third-party
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np  # Fallback to numpy if cupy not available
    CUPY_AVAILABLE = False

# Local config
from .pathfinder.config import PAD_CLEARANCE_MM
from .pad_escape_planner import PadEscapePlanner, Portal
from .hdi_stack import HDIStack, canonical_pair
from .board_analyzer import (
    analyze_board_characteristics,
    BoardCharacteristics,
    preferred_layer_directions_for_board,
)
from .parameter_derivation import derive_routing_parameters, apply_derived_parameters, DerivedRoutingParameters
from .pathfinder.via_kernels import ViaKernelManager, convert_via_metadata_to_gpu, ensure_gpu_array

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


class GPUConfig:
    """GPU pathfinding algorithm configuration"""
    GPU_MODE = True  # Enable GPU acceleration (set to False for CPU-only)
    DEBUG_INVARIANTS = True
    USE_PERSISTENT_KERNEL = False  # DISABLED: Hangs on cooperative kernel launch; using wavefront with atomic keys instead
    USE_GPU_COMPACTION = True
    USE_DELTA_STEPPING = False  # DISABLED: Causes OOM from bucket allocation on large graphs
    DELTA_VALUE = 0.5  # Bucket width in mm (0.5mm ≈ 1.25 × 0.4mm grid pitch)
    # Recommended delta values:
    # - 0.4: Same as grid pitch (many buckets, high precision)
    # - 0.5: 1.25× grid pitch (good balance) ← DEFAULT
    # - 0.8: 2× grid pitch (fewer buckets, faster but less precise)
    # - 1.6: 4× grid pitch (degenerates toward Dijkstra)


class PathFinderConfig:
    """PathFinder algorithm parameters - TUNED FOR FAST CONVERGENCE

    Core convergence mechanism: Pathfinder uses negotiated congestion routing where
    edge costs increase based on usage (present) and history. The cost function is:
        total_cost = base + pres_fac*overuse + hist_gain*history

    Key parameters for tuning:
    - pres_fac: Present congestion penalty (increases each iteration)
    - hist_gain: Historical congestion penalty (permanent after each iteration)
    - via_cost: Vertical movement penalty (lower = more layer exploration)

    WARNING: Cost function modifications are HIGH-RISK. Small changes can cause
    20%+ convergence regression. Prefer infrastructure improvements over tuning.
    """
    max_iterations: int = 40  # Extended to give convergence more time
    # Ownership-as-cost: entering a node owned by another net costs
    # owner_penalty_base * pres_fac instead of being hard-forbidden, so
    # barrel conflicts are negotiated like any other congestion.
    owner_penalty_base: float = 25.0
    # A via inserted after a planar route cannot see that track in the
    # via-only ownership map. Price every occupied path node equally so
    # track-via conflicts are symmetric within the same routing pass.
    path_node_penalty_base: float = 25.0
    # Coordinates that produced a physical node/barrel short retain a
    # Pathfinder history cost after their present owner is ripped up.
    node_history_penalty: float = 5.0
    node_history_increment: float = 1.0
    # CONVERGENCE SCHEDULE (BALANCED BASED ON DIAGNOSTICS):
    pres_fac_init: float = 1.0   # Start gentle (iteration 1)
    pres_fac_mult: float = 1.10  # Gentler exponential to keep history competitive (was 1.15)
    pres_fac_max: float = 64.0    # Allow higher pressure to resolve late-stage congestion (was 8.0)
    hist_gain: float = 0.2       # Lowered for raw present (was 0.8 with present_ema)

    # CRITICAL: Length vs Completion Trade-off
    base_cost_weight: float = 0.3  # Weight for path length penalty (1.0=optimize length, 0.01=optimize completion)
    # Setting this to 0.01-0.1 makes router prefer completion over short paths,
    # enabling use of empty vertical channels. Lower values = more detours, higher completion.

    grid_pitch: float = 0.4
    # Preferred planar axis for each copper layer. None preserves the
    # historical F.Cu=V, then alternating H/V assignment. A finite
    # wrong_way_cost_multiplier adds the nonpreferred axis to the graph:
    # >1 is guided routing, 1 is fully bidirectional, and infinity is strict.
    preferred_layer_directions: Optional[List[str]] = None
    wrong_way_cost_multiplier: float = float("inf")
    # Additive multiplicative-cost bias per layer index. Zero is neutral;
    # positive values pack routes toward lower layers while leaving every
    # layer available when congestion makes spilling upward worthwhile.
    layer_depth_bias: float = 0.0
    # Physical output geometry. Parsed KiCad project rules replace these
    # defaults during initialize_graph().
    track_width: float = 0.24
    clearance: float = 0.15
    via_diameter: float = 0.25
    via_drill: float = 0.15
    min_hole_to_hole: float = 0.25
    hole_clearance: float = 0.1
    via_cost: float = 0.7  # Cheaper vias to encourage layer hopping and redistribute load (was 1.0)
    via_pressure_threshold: int = 64
    via_pressure_multiplier: float = 1.5
    portal_discount: float = 0.4  # 60% discount on first escape via from terminals
    span_alpha: float = 0.15  # Span penalty: cost *= (1 + alpha*(span-1))
    # None selects full spans through 18 layers and adjacent spans on deeper
    # stacks where the O(L²) graph is prohibitive.
    allow_any_layer_via: Optional[bool] = None
    # Adjacent graph hops represent segments of one physical multi-layer via.
    adjacent_via_step_scale: float = 4.0

    # Iteration 1 policy: always-connect mode for maximum connectivity
    iter1_always_connect: bool = True  # Use soft costs in iteration 1 instead of hard blocks

    # Portal escape configuration
    portal_enabled: bool = True
    portal_delta_min: int = 3      # Min vertical offset (1.2mm @ 0.4mm pitch)
    portal_delta_max: int = 12     # Max vertical offset (4.8mm)
    portal_candidate_delta_min: int = 1
    portal_delta_pref: int = 6     # Preferred offset (2.4mm)
    portal_x_snap_max: float = 0.5  # Max x-snap in steps (½ pitch)
    portal_via_discount: float = 0.15  # Escape via multiplier (85% discount)
    portal_retarget_patience: int = 3  # Iters before retargeting
    portal_candidate_count: int = 20
    escape_assignment_steps: int = 5000
    escape_reservation_penalty: float = 1000.0
    escape_preference_penalty: float = 10.0
    escape_replan_patience: int = 2
    escape_replan_limit: int = 3
    # A portal barrel that repeatedly blocks another routed net must move.
    # Accumulated history prevents the owner from returning to the same
    # locally attractive but globally impossible escape column.
    portal_barrel_history_penalty: float = 25.0
    portal_cleanup_edge_penalty: float = 1_000_000.0
    portal_cleanup_node_penalty: float = 1_000_000.0
    portal_cleanup_escape_penalty: float = 1_000_000.0
    portal_cleanup_edge_threshold: int = 3

    stagnation_patience: int = 5
    # A strictly-new minimum is not sufficient evidence of useful progress
    # on a monster route. Measure reduction across a rolling window and
    # temporarily widen severe hotsets when the fractional descent is too
    # small. Repeated slow windows then raise the pressure ceiling in stages.
    slow_progress_window: int = 5
    # Five passes must remove roughly 5/200 of the starting residual to stay
    # on a 200-pass linear clearing pace. The tail has separate smaller-wave
    # policy, so apply this only while severe congestion remains.
    slow_progress_min_fraction: float = 0.025
    slow_progress_min_overuse: int = 16_384
    slow_progress_hotset_cap: int = 512
    # The first measured 512-net severe window cleared 1.546x more excess per
    # bounded iteration than its matched 256-net window while remaining
    # essentially wall-clock neutral. A later full-board 1024-net window
    # cleared 2.466x more per pass than the immediately preceding 256-net
    # plateau at 0.871x wall efficiency, with no negotiated or exact-physical
    # debt category diverging. Continue the measured doubling ladder once
    # more on a third independently separated plateau.
    slow_progress_hotset_growth_after: int = 2
    slow_progress_hotset_cap_max: int = 2048
    slow_progress_pressure_after: int = 2
    slow_progress_pres_fac_max: float = 256.0
    # Selective PathFinder passes do unequal work. Advance pressure by a
    # bounded equivalent number of the historical 100-net reroute waves so a
    # 256/512-net pass does not delay the pressure schedule in wall time.
    pressure_reference_hotset: int = 100
    pressure_work_scale_max: float = 2.0
    use_gpu: bool = True  # GPU algorithm fixed, validation will catch ROI construction issues
    batch_size: int = 32
    layer_count: int = 6
    strict_drc: bool = False  # Legacy compatibility
    mode: str = "near_far"
    roi_parallel: bool = False
    per_net_budget_s: float = 5.0
    max_roi_nodes: int = 750000  # Increased from 500K to 750K to accommodate large inter-bank nets
    delta_multiplier: float = 4.0
    adaptive_delta: bool = True
    strict_capacity: bool = True
    live_present_costs: bool = True
    reroute_only_offenders: bool = True
    # Complete production routes may not silently drop difficult nets.
    # Retain the legacy quarantine only as an opt-in diagnostic.
    allow_net_exclusion: bool = False
    layer_shortfall_percentile: float = 95.0
    layer_shortfall_cap: int = 16
    enable_profiling: bool = False
    enable_instrumentation: bool = False
    strict_overuse_block: bool = False
    hist_cost_weight: float = 10.0  # Boost history weight to compete with present (was 2.0)
    log_iteration_details: bool = False
    acc_fac: float = 0.0
    phase_block_after: int = 2
    congestion_multiplier: float = 1.0
    max_search_nodes: int = 2000000
    # Avoid heap Dijkstra over an entire multi-million-node graph after CUDA
    # has already failed for the current negotiated cost state.
    gpu_fullgraph_fail_fast_nodes: int = 1_000_000
    layer_names: List[str] = field(default_factory=lambda: ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu'])
    # Absolute negotiation-wave ceiling. Normal severe passes remain 256 via
    # _history_hotset_cap(); measured slow windows may use 512, 1024, then
    # 2048.
    # Absolute safety ceiling for the measured 512 -> 1024 -> 2048 severe
    # plateau ladder. Ordinary severe passes still select only 256 nets.
    hotset_cap: int = 2048
    # Physical conflicts may implicate nearly every net on a monster board.
    # Rerouting that near-global set after graph congestion is already low
    # destroys the best-known solution. Process the worst physical offenders
    # in bounded waves; boards at or below this size remain unchanged.
    physical_hotset_cap: int = 1024
    physical_hotset_min: int = 64
    physical_conflicts_per_hotset_net: float = 50.0
    # Stable keepers prevent via-pool peers from swapping ownership every
    # pass. Permit rotation only in a board-scaled tail: eight unresolved
    # resources per 1,024 routed nets preserves the small-board policy while
    # allowing a monster route's independent tails to make progress.
    via_keeper_rotation_overuse_threshold: int = 8
    via_keeper_rotation_nets_per_step: int = 1024
    allowed_via_spans: Optional[Set[Tuple[int, int]]] = None  # None = all layer pairs allowed (blind/buried)
    # Explicit fabrication topology. When set, graph vias and emitted copper
    # retain the stack's legal spans and span-specific drill processes.
    hdi_stack: Optional[HDIStack] = None


# Legacy constants
DEFAULT_GRID_PITCH = 0.4
GRID_PITCH = 0.4
LAYER_COUNT = 6


def resolve_history_decay(config) -> float:
    """Return the derived history-retention factor, with a test override."""
    value = float(os.getenv(
        "ORTHO_HISTORY_DECAY",
        getattr(config, "history_decay", 1.0),
    ))
    if not 0.0 <= value <= 1.0:
        raise ValueError("ORTHO_HISTORY_DECAY must be between 0.0 and 1.0")
    return value


def resolve_pres_fac_max(config, signal_layers: int) -> float:
    """Respect the configured ceiling while enforcing a routing-size floor."""
    if signal_layers <= 12:
        layer_floor = 32.0
    elif signal_layers <= 20:
        layer_floor = 64.0
    else:
        layer_floor = 128.0
    configured = float(getattr(config, "pres_fac_max", layer_floor))
    if not np.isfinite(configured) or configured <= 0:
        raise ValueError(
            f"pres_fac_max must be finite and positive, got {configured!r}"
        )
    return max(configured, layer_floor)


# ═══════════════════════════════════════════════════════════════════════════════
# CSR GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def _points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test, vectorised over N points.

    (From PR #17 by RolandWa.)

    Args:
        px, py: 1-D float arrays of query point coordinates (mm), shape (N,)
        poly:   polygon vertices, shape (M, 2) in (x, y) order

    Returns:
        Boolean array of shape (N,), True if the point is inside the polygon.
    """
    n = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        cond1 = (yi > py) != (yj > py)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_intersect = np.where(
                cond1,
                (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi,
                np.inf
            )
        inside ^= cond1 & (px < x_intersect)
        j = i
    return inside


class CSRGraph:
    """Compressed Sparse Row graph"""

    def __init__(self, use_gpu=False, edge_capacity=None):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.indptr = None
        self.indices = None
        self.base_costs = None

        # Pre-allocate numpy array if capacity known (memory efficient).
        # NOTE: must test against None, not truthiness - a capacity of 0 is
        # a real (degenerate) value and previously fell into the list path,
        # producing a confusing bare "No edges" error.
        if edge_capacity is not None:
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

    def finalize(self, num_nodes: int, num_layers: int = 0):
        """Build CSR from edge list (memory-efficient)"""
        import time
        start = time.time()

        E = self._edge_idx if self._use_array else len(self._edges)
        if E == 0:
            raise ValueError(
                "Routing graph has no edges. This usually means no routing "
                "layers were derived from the board (check layer_count and "
                "lattice bounds).")

        if self._use_array:
            # Already in numpy array (pre-allocated)
            edge_array = self._edge_array[:E]  # Trim to actual size
            logger.info(f"Finalizing CSR: {E:,} edges from pre-allocated array")
        else:

            # Convert to numpy array for memory-efficient sorting
            logger.info(f"Converting {E:,} edges to numpy array...")
            edge_array = np.array(self._edges, dtype=[('src', 'i4'), ('dst', 'i4'), ('cost', 'f4')])
            # Free memory immediately
            self._edges.clear()

        # Sort by source node - GPU accelerated if available!
        logger.info(f"Sorting {E:,} edges by source node...")
        sort_start = time.time()

        # OPTIMIZATION: Use GPU sort if available (8-10× faster for large arrays)
        if self.use_gpu and GPU_AVAILABLE:
            try:
                logger.info(f"[GPU-SORT] Using CuPy GPU radix sort (expected ~3-5s for 54M edges)")
                # Extract 'src' field as contiguous array (CuPy doesn't support structured arrays)
                src_keys = edge_array['src'].copy()
                # Transfer just the sort keys to GPU
                src_keys_gpu = cp.asarray(src_keys)
                # GPU radix sort to get indices (much faster than CPU quicksort/mergesort)
                sorted_idx = cp.argsort(src_keys_gpu, kind='stable')
                # Transfer indices back to CPU
                sorted_idx_cpu = sorted_idx.get()
                # Reorder the structured array using GPU-computed indices
                edge_array = edge_array[sorted_idx_cpu]
                sort_time = time.time() - sort_start
                logger.info(f"[GPU-SORT] GPU sort completed in {sort_time:.1f} seconds ({E/sort_time/1e6:.1f}M edges/sec)")
            except Exception as e:
                logger.warning(f"[GPU-SORT] GPU sort failed: {e}, falling back to CPU")
                # CPU fallback
                edge_array.sort(order='src', kind='mergesort')
                sort_time = time.time() - sort_start
                logger.info(f"[CPU-SORT] CPU sort completed in {sort_time:.1f} seconds")
        else:
            # CPU sort (stable mergesort)
            edge_array.sort(order='src', kind='mergesort')
            sort_time = time.time() - sort_start
            logger.info(f"[CPU-SORT] Sort completed in {sort_time:.1f} seconds")

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

        # Build edge-to-layer mapping for layer balancing (vectorized, one-time cost)
        # CRITICAL: Must do this BEFORE transferring to GPU, while indptr is still on CPU
        if num_layers > 0:
            plane_size = num_nodes // num_layers
            # For each edge, determine its layer from source node
            # Vectorized approach: build array of source nodes for each edge
            edge_sources = np.zeros(E, dtype=np.int32)
            edge_targets = np.zeros(E, dtype=np.int32)
            for u in range(num_nodes):
                start, end = indptr[u], indptr[u+1]
                if end > start:
                    edge_sources[start:end] = u
                    edge_targets[start:end] = indices[start:end]

            # Compute layer for each edge: layer = source_node // plane_size
            self.edge_layer = (edge_sources // plane_size).astype(np.uint8)

            # Compute edge_kind: 0 = horizontal/vertical (same layer), 1 = via (different layers)
            source_layers = edge_sources // plane_size
            target_layers = edge_targets // plane_size
            self.edge_kind = (source_layers != target_layers).astype(np.uint8)

            via_count = int(np.sum(self.edge_kind))
            horiz_vert_count = E - via_count
            logger.info(f"[LAYER-MAP] Built edge→layer mapping: {E} edges, {num_layers} layers")
            logger.info(f"[EDGE-KIND] Horizontal/Vertical={horiz_vert_count}, Via={via_count}")
        else:
            self.edge_layer = None
            self.edge_kind = None

        if self.use_gpu:
            self.indptr = cp.asarray(indptr)
            self.indices = cp.asarray(indices)
            self.base_costs = cp.asarray(costs)
            # Transfer edge_layer and edge_kind to GPU if they exist
            if self.edge_layer is not None:
                self.edge_layer_gpu = cp.asarray(self.edge_layer)
            if self.edge_kind is not None:
                self.edge_kind_gpu = cp.asarray(self.edge_kind)
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
        self.present_ema = self.xp.zeros(num_edges, dtype=self.xp.float32)  # Smoothed present for stable convergence
        self.history = self.xp.zeros(num_edges, dtype=self.xp.float32)
        self.capacity = self.xp.ones(num_edges, dtype=self.xp.float32)
        self.total_cost = None

    def cost_balance_ratio(
        self,
        history_weight: float,
        present_factor: float,
    ) -> float:
        """Compare the history and present terms actually used in costs."""
        present_overuse = self.xp.maximum(
            0, self.present_ema - self.capacity
        )
        history_cost = float(history_weight * self.history.sum())
        present_cost = float(
            present_factor * present_overuse.sum()
        )
        return history_cost / (present_cost + 1e-9)

    @property
    def edge_usage(self):
        """Compatibility view of canonical sparse edge occupancy."""
        return self.canonical

    def refresh_from_canonical(self):
        """Rebuild present"""
        self.present.fill(0)
        if not self.canonical:
            return
        indices = np.fromiter(
            self.canonical.keys(),
            dtype=np.int64,
            count=len(self.canonical),
        )
        counts = np.fromiter(
            self.canonical.values(),
            dtype=np.float32,
            count=len(self.canonical),
        )
        valid = (indices >= 0) & (indices < self.E)
        if not valid.all():
            indices = indices[valid]
            counts = counts[valid]
        device_indices = self.xp.asarray(indices)
        self.present[device_indices] = self.xp.asarray(counts)

    def commit_path(self, edge_indices: List[int]):
        """Add path and keep present in sync.

        Vectorized scatter-add on present (from PR #17 by RolandWa): the old
        per-element loop was one host<->device round-trip per edge on GPU.
        The canonical dict stays a CPU loop - fast for typical path lengths.
        """
        if not edge_indices:
            return
        for idx in edge_indices:
            self.canonical[idx] = self.canonical.get(idx, 0) + 1
        arr = self.xp.asarray(edge_indices, dtype=np.int64)
        if self.use_gpu:
            import cupyx
            cupyx.scatter_add(self.present, arr,
                              self.xp.ones(len(edge_indices), dtype=self.present.dtype))
        else:
            np.add.at(self.present, arr, 1.0)

    def clear_path(self, edge_indices: List[int]):
        """Remove path and keep present in sync (vectorized, floored at 0)."""
        if not edge_indices:
            return
        for idx in edge_indices:
            if idx in self.canonical:
                self.canonical[idx] -= 1
                if self.canonical[idx] <= 0:
                    del self.canonical[idx]
        arr = self.xp.asarray(edge_indices, dtype=np.int64)
        if self.use_gpu:
            import cupyx
            cupyx.scatter_add(self.present, arr,
                              -self.xp.ones(len(edge_indices), dtype=self.present.dtype))
            self.xp.maximum(self.present, 0, out=self.present)
        else:
            np.add.at(self.present, arr, -1.0)
            np.maximum(self.present, 0, out=self.present)

    def compute_overuse(self, router_instance=None) -> Tuple[int, int]:
        """
        Compute physical-resource overuse including edge and via pools.

        Paths reserve both directed CSR arcs of each physical segment so
        opposite traversals collide. When a router is available, count only
        one canonical arc per segment; otherwise one physical edge violation
        would be weighted twice relative to a capacity-one path node.

        Args:
            router_instance: Optional PathFinderRouter instance for via spatial checks

        Returns:
            (total_overuse_sum, edge_overuse_count)
        """
        # Edge overuse (existing)
        usage = self.present.get() if self.use_gpu else self.present
        cap = self.capacity.get() if self.use_gpu else self.capacity
        edge_over = np.maximum(0, usage - cap)
        if (
            router_instance is not None
            and hasattr(
                router_instance,
                "_canonical_edge_resource_mask",
            )
        ):
            edge_over = edge_over[
                router_instance._canonical_edge_resource_mask()
            ]
        edge_over_sum = int(edge_over.sum())
        edge_over_count = int(np.count_nonzero(edge_over))

        # Via spatial overuse (NEW)
        via_col_over_sum = 0
        via_seg_over_sum = 0

        if router_instance is not None:
            # Check via column overuse
            if hasattr(router_instance, 'via_col_use') and hasattr(router_instance, 'via_col_cap'):
                via_col_over = np.maximum(0, router_instance.via_col_use - router_instance.via_col_cap)
                via_col_over_sum = int(via_col_over.sum())

            # Check via segment overuse
            if hasattr(router_instance, 'via_seg_use') and hasattr(router_instance, 'via_seg_cap'):
                via_seg_over = np.maximum(0, router_instance.via_seg_use - router_instance.via_seg_cap)
                via_seg_over_sum = int(via_seg_over.sum())

        total_over = edge_over_sum + via_col_over_sum + via_seg_over_sum

        # Log via violations if present (helps with debugging)
        if via_col_over_sum > 0 or via_seg_over_sum > 0:
            logger.info(f"[OVERUSE] edge={edge_over_sum} via_col={via_col_over_sum} via_seg={via_seg_over_sum} total={total_over}")

        return (total_over, edge_over_count)

    def verify_present_matches_canonical(self) -> bool:
        """Sanity check: verify present usage matches canonical store"""
        recomputed = self.xp.zeros(self.E, dtype=self.xp.float32)
        if self.canonical:
            indices = np.fromiter(
                self.canonical.keys(),
                dtype=np.int64,
                count=len(self.canonical),
            )
            counts = np.fromiter(
                self.canonical.values(),
                dtype=np.float32,
                count=len(self.canonical),
            )
            valid = (indices >= 0) & (indices < self.E)
            if not valid.all():
                indices = indices[valid]
                counts = counts[valid]
            device_indices = self.xp.asarray(indices)
            recomputed[device_indices] = self.xp.asarray(counts)

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

    def update_history(self, gain: float, base_costs=None, history_cap_multiplier=10.0, decay_factor=0.98, use_raw_present=False):
        """
        Update history with:
        - Gentle decay: history *= 0.98 before adding increment (decay_factor param)
        - Clamping: increment capped at history_cap = 10 * base_cost
        - Uses present_ema (smoothed) by default, or raw present if use_raw_present=True
        """
        import logging
        import sys
        logger = logging.getLogger(__name__)

        # DIAGNOSTIC: Log what's actually happening
        if not hasattr(self, '_hist_update_count'):
            self._hist_update_count = 0
        self._hist_update_count += 1

        # Always log first 5 calls
        if self._hist_update_count <= 5:
            # Before update
            hist_before_max = float(self.history.max()) if self.history.size > 0 else 0.0
            logger.debug(f"[UPDATE-HISTORY CALLED] Call #{self._hist_update_count} START gain={gain:.3f}")

        # Apply gentle decay before adding new history
        self.history *= decay_factor

        # Use smoothed present_ema by default, or raw present if requested
        present_for_history = self.present if use_raw_present else self.present_ema
        over = self.xp.maximum(0, present_for_history - self.capacity)
        increment = gain * over

        # Clamp per-edge history increment
        if base_costs is not None:
            history_cap = history_cap_multiplier * base_costs
            increment_before_cap = increment.copy()
            increment = self.xp.minimum(increment, history_cap)

            if self._hist_update_count <= 5:
                # Check how many edges are being capped
                capped_mask = increment_before_cap > history_cap
                capped_count = int(self.xp.sum(capped_mask))
                if capped_count > 0:
                    logger.debug(f"  [HIST-CAP] {capped_count} edges capped! avg_cap={float(history_cap.mean()):.3f}")

        self.history += increment

        if self._hist_update_count <= 5:
            # After update
            hist_after_max = float(self.history.max())
            incr_max = float(increment.max())
            over_max = float(over.max())
            over_mean = float(over[over > 0].mean()) if (over > 0).any() else 0.0
            pres_max = float(present_for_history.max())
            pres_ema_max = float(self.present_ema.max())
            pres_raw_max = float(self.present.max())

            logger.debug(f"[UPDATE-HISTORY #{self._hist_update_count}]")
            logger.debug(f"  gain={gain:.3f} decay={decay_factor:.3f} cap_mult={history_cap_multiplier:.1f}")
            logger.debug(f"  use_raw_present={use_raw_present}")
            logger.debug(f"  present_raw_max={pres_raw_max:.1f} present_ema_max={pres_ema_max:.1f}")
            logger.debug(f"  overuse: max={over_max:.2f} mean={over_mean:.3f}")
            logger.debug(f"  increment: max={incr_max:.3f}")
            logger.debug(f"  history: before={hist_before_max:.3f} → after={hist_after_max:.3f}")
            if base_costs is not None:
                logger.debug(f"  base_cost: mean={float(base_costs.mean()):.4f} max={float(base_costs.max()):.4f}")

    def update_present_ema(self, beta: float = 0.60):
        """
        Update exponential moving average of present usage for stability.
        Smooths bang-bang oscillations in overuse detection.

        Args:
            beta: EMA smoothing factor (higher = more smoothing, typically 0.6)
        """
        self.present_ema = beta * self.present + (1.0 - beta) * self.present_ema

    def update_costs(
        self,
        base_costs,
        pres_fac: float,
        hist_weight: float = 1.0,
        add_jitter: bool = True,
        via_cost_multiplier: float = 1.0,
        base_cost_weight: float = 0.01,
        *,
        edge_layer=None,          # np/cp array [E] with source layer per edge
        layer_bias_per_layer=None,  # np/cp array [L] with multiplicative bias
        edge_kind=None            # np/cp array [E] with 0=horiz/vert, 1=via
    ):
        """
        total = (base * via_multiplier * base_weight * layer_bias) + pres_fac*overuse + hist_weight*history + epsilon_jitter
        Jitter breaks ties and prevents oscillation in equal-cost paths.
        Via cost multiplier enables late-stage via annealing.
        Base cost weight controls length vs completion trade-off (lower = prefer completion over short paths).
        Layer bias: applied only to horizontal/vertical edges (not vias) to rebalance layer usage.
        Uses present_ema (smoothed) instead of raw present to prevent bang-bang oscillation.
        """
        xp = self.xp
        # Use smoothed present (EMA) to prevent oscillation - critical for convergence
        over = xp.maximum(0, self.present_ema - self.capacity)

        # Vectorized per-edge layer bias (single gather operation)
        # Only apply to horizontal/vertical edges (edge_kind==0), not vias (edge_kind==1)
        per_edge_bias = 1.0
        if (edge_layer is not None) and (layer_bias_per_layer is not None) and (edge_kind is not None):
            if self.use_gpu:
                # Ensure arrays are on GPU
                layer_bias = cp.asarray(layer_bias_per_layer) if not hasattr(layer_bias_per_layer, "get") else layer_bias_per_layer
                edge_layer_arr = cp.asarray(edge_layer) if not hasattr(edge_layer, "get") else edge_layer
                edge_kind_arr = cp.asarray(edge_kind) if not hasattr(edge_kind, "get") else edge_kind
            else:
                # NumPy arrays
                layer_bias = layer_bias_per_layer
                edge_layer_arr = edge_layer
                edge_kind_arr = edge_kind

            # Gather bias for each edge's layer
            bias_factors = layer_bias[edge_layer_arr]
            # Apply bias only to horizontal/vertical edges (edge_kind==0), set via edges to 1.0
            per_edge_bias = xp.where(edge_kind_arr == 0, bias_factors, 1.0)

        # Apply both via multiplier, base weight, and layer bias to base costs
        # base_cost_weight < 1.0 makes router prefer completion over short paths
        adjusted_base = base_costs * via_cost_multiplier * base_cost_weight * per_edge_bias

        # Apply INVERTED layer bias to present term to directly pressure hot layers
        # Base term uses per_edge_bias (hot layers cheaper for length optimization)
        # Present term uses INVERSE (hot layers more expensive for congestion avoidance)
        # For vias, keep bias at 1.0 (no layer-specific present penalty)
        if (edge_layer is not None) and (layer_bias_per_layer is not None) and (edge_kind is not None):
            # Invert bias for present: if bias=0.9 (cheap base), use 1/0.9=1.11 (expensive present)
            # Clamp to prevent extreme values
            inverted_bias = xp.where(per_edge_bias != 0, 1.0 / xp.maximum(per_edge_bias, 0.5), 1.0)
            inverted_bias = xp.where(edge_kind_arr == 0, inverted_bias, 1.0)  # Only H/V edges
            present_term = (pres_fac * inverted_bias) * over
            self._present_cost_scale = pres_fac * inverted_bias
        else:
            present_term = pres_fac * over
            self._present_cost_scale = float(pres_fac)

        self.total_cost = adjusted_base + present_term + hist_weight * self.history

        # Add per-edge epsilon jitter to break ties (stable across iterations)
        if add_jitter:
            E = len(self.total_cost)
            # Use edge index modulo prime for deterministic jitter
            jitter = xp.arange(E, dtype=xp.float32) % 9973
            jitter = jitter * 1e-6  # tiny epsilon
            self.total_cost += jitter

    def begin_live_present_costs(self):
        """Switch from iteration EMA to prospective per-net occupancy costs."""
        scale = getattr(self, "_present_cost_scale", 1.0)
        iteration_over = self.xp.maximum(
            0, self.present_ema - self.capacity
        )
        # Preserve base, history, jitter, pooling penalties, and hard blocks.
        self._live_static_cost = (
            self.total_cost - scale * iteration_over
        )
        prospective_over = self.xp.maximum(
            0, self.present + 1.0 - self.capacity
        )
        self.total_cost[:] = (
            self._live_static_cost + scale * prospective_over
        )

    def refresh_live_present_costs(self, edge_indices: List[int]):
        """Reprice touched edges after one net is ripped up or committed."""
        if not edge_indices or not hasattr(self, "_live_static_cost"):
            return
        edge_array = self.xp.asarray(edge_indices, dtype=np.int64)
        scale = getattr(self, "_present_cost_scale", 1.0)
        prospective_over = self.xp.maximum(
            0,
            self.present[edge_array]
            + 1.0
            - self.capacity[edge_array],
        )
        if np.isscalar(scale):
            edge_scale = scale
        else:
            edge_scale = scale[edge_array]
        self.total_cost[edge_array] = (
            self._live_static_cost[edge_array]
            + edge_scale * prospective_over
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3D LATTICE
# ═══════════════════════════════════════════════════════════════════════════════

class Lattice3D:
    """3D routing lattice with configurable preferred H/V discipline."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        pitch: float,
        layers: int,
        preferred_layer_directions: Optional[List[str]] = None,
        wrong_way_cost_multiplier: float = float("inf"),
    ):
        self.bounds = bounds
        self.pitch = pitch
        self.layers = layers
        self.preferred_layer_directions = (
            list(preferred_layer_directions)
            if preferred_layer_directions is not None
            else None
        )
        self.wrong_way_cost_multiplier = float(
            wrong_way_cost_multiplier
        )
        if (
            self.wrong_way_cost_multiplier < 1.0
            or (
                not np.isfinite(self.wrong_way_cost_multiplier)
                and self.wrong_way_cost_multiplier != float("inf")
            )
        ):
            raise ValueError(
                "wrong_way_cost_multiplier must be >= 1 or infinity"
            )

        self.geom = KiCadGeometry(bounds, pitch, layer_count=layers)
        self.x_steps = self.geom.x_steps
        self.y_steps = self.geom.y_steps
        self.num_nodes = self.x_steps * self.y_steps * layers

        self.layer_dir = self._assign_directions()
        logger.info(f"Lattice: {self.x_steps}×{self.y_steps}×{layers} = {self.num_nodes:,} nodes")

    def _assign_directions(self) -> List[str]:
        """F.Cu=V (vertical escape routing), internal layers alternate H/V"""
        if self.preferred_layer_directions is not None:
            if len(self.preferred_layer_directions) != self.layers:
                raise ValueError(
                    "preferred_layer_directions must have one H/V entry "
                    f"per layer ({self.layers} expected)"
                )
            directions = [
                str(axis).strip().lower()
                for axis in self.preferred_layer_directions
            ]
            if any(axis not in ("h", "v") for axis in directions):
                raise ValueError(
                    "preferred_layer_directions entries must be H or V"
                )
            return directions
        dirs = []
        for z in range(self.layers):
            if z == 0:
                # F.Cu has vertical routing for escape stubs
                dirs.append('v')
            else:
                # Internal layers alternate: In1.Cu=H, In2.Cu=V, In3.Cu=H, etc.
                dirs.append('h' if z % 2 == 1 else 'v')
        return dirs

    def get_legal_axis(self, layer: int) -> str:
        """Return the preferred planar axis for one layer."""
        if layer >= len(self.layer_dir):
            return 'h' if layer % 2 == 1 else 'v'
        return self.layer_dir[layer]

    def get_allowed_axes(self, layer: int) -> Tuple[str, ...]:
        """Return planar axes materialized on one layer."""
        preferred = self.get_legal_axis(layer)
        if np.isfinite(self.wrong_way_cost_multiplier):
            other = "v" if preferred == "h" else "h"
            return (preferred, other)
        return (preferred,)

    def planar_cost_multiplier(self, layer: int, axis: str) -> float:
        """Return the preferred/wrong-way cost multiplier for an axis."""
        normalized = str(axis).lower()
        if normalized not in self.get_allowed_axes(layer):
            return float("inf")
        if normalized == self.get_legal_axis(layer):
            return 1.0
        return self.wrong_way_cost_multiplier

    def is_legal_planar_edge(self, from_x: int, from_y: int, from_layer: int,
                              to_x: int, to_y: int, to_layer: int) -> bool:
        """Check if planar edge follows H/V discipline."""
        if from_layer != to_layer:
            return True  # Vias always legal (checked separately)

        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)

        # Must be adjacent (Manhattan distance 1)
        if dx + dy != 1:
            return False

        axis = "h" if dx == 1 else "v"
        return axis in self.get_allowed_axes(from_layer)

    def get_legal_via_pairs(
        self, layer_count: int, allow_any_layer_via: bool = False
    ) -> set:
        """
        Return canonical unordered legal via layer pairs.

        ``build_graph`` materializes both directed edges for every pair, so
        returning both ``(a, b)`` and ``(b, a)`` would duplicate each graph
        edge. Keep the lower layer first to make that invariant explicit.

        CRITICAL: Must include F.Cu (layer 0) → internal layer transitions!
        The escape planner creates stubs on F.Cu, and PathFinder must be able
        to create vias from F.Cu to whatever internal layer it chooses.
        """
        if layer_count <= 2:
            # 2-layer board: there are no inner layers, so F.Cu/B.Cu are the
            # routing layers and the only via is the (0,1) through-hole.
            logger.info("[VIA-PAIRS] 2-layer board: through vias only")
            return {(0, 1)}

        # Internal routing layers (exclude B.Cu which is layer_count-1)
        routing_layers = list(range(1, layer_count - 1))

        logger.info(f"[VIA-PAIRS] layer_count={layer_count}, routing_layers={len(routing_layers)}, "
                   f"allow_any={allow_any_layer_via}")

        if allow_any_layer_via:
            # FULL BLIND/BURIED: Any routing layer to any other routing layer
            legal_pairs = set()
            for z1 in routing_layers:
                for z2 in routing_layers:
                    if z1 < z2:
                        legal_pairs.add((z1, z2))

            # CRITICAL: Add F.Cu (layer 0) → internal layer transitions
            # This allows PathFinder to create escape vias from F.Cu to any internal layer
            for z in routing_layers:
                legal_pairs.add((0, z))

            logger.info(
                f"[VIA-PAIRS] Generated {len(legal_pairs)} unordered pairs; "
                "build_graph emits both directions"
            )
            return legal_pairs

        # O(L) representation: a physical long via is a chain of adjacent
        # segments. Include F.Cu→In1.Cu, but never B.Cu.
        legal_pairs = {
            (z, z + 1) for z in range(0, layer_count - 2)
        }
        logger.info(
            f"[VIA-PAIRS] Generated {len(legal_pairs)} adjacent-only pairs"
        )
        return legal_pairs

    def node_idx(self, x: int, y: int, z: int) -> int:
        """(x,y,z) → flat"""
        return self.geom.node_index(x, y, z)

    def idx_to_coord(self, idx: int) -> Tuple[int, int, int]:
        """flat → (x,y,z)"""
        return self.geom.index_to_coords(idx)

    def world_to_lattice(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """mm → lattice"""
        return self.geom.world_to_lattice(x_mm, y_mm)

    @staticmethod
    def _via_site_policy(required_spacing: float, pitch: float):
        """Return the sublattice policy satisfying one center spacing."""
        required_spacing = max(0.0, float(required_spacing))
        if required_spacing < pitch - 1e-9:
            return "all", 1
        if required_spacing <= pitch * np.sqrt(2.0) + 1e-9:
            return "checkerboard", 1
        return "stride", max(2, int(np.ceil(required_spacing / pitch)))

    @staticmethod
    def _matches_via_site_policy(
        x: int, y: int, mode: str, stride: int
    ) -> bool:
        if mode == "all":
            return True
        if mode == "checkerboard":
            return (x + y) % 2 == 0
        return x % stride == 0 and y % stride == 0

    def is_via_site(
        self,
        x: int,
        y: int,
        z_from: Optional[int] = None,
        z_to: Optional[int] = None,
    ) -> bool:
        """Return whether one via process may occupy this lattice site."""
        if z_from is not None and z_to is not None:
            pair = canonical_pair(int(z_from), int(z_to))
            policies = getattr(self, "_via_pair_site_policies", {})
            mode, stride = policies.get(
                pair,
                (
                    getattr(self, "_via_site_mode", "all"),
                    int(getattr(self, "_via_site_stride", 1)),
                ),
            )
        else:
            mode = getattr(self, "_via_site_mode", "all")
            stride = int(getattr(self, "_via_site_stride", 1))
        return self._matches_via_site_policy(x, y, mode, stride)

    def build_graph(
        self,
        via_cost: float,
        allowed_via_spans: Optional[Set[Tuple[int, int]]] = None,
        use_gpu=False,
        allow_any_layer_via: bool = False,
        adjacent_via_step_scale: float = 4.0,
        min_via_center_spacing: float = 0.0,
        via_pair_center_spacing: Optional[
            Dict[Tuple[int, int], float]
        ] = None,
    ) -> CSRGraph:
        """
        Build graph with H/V constraints and flexible via spans.

        Args:
            via_cost: Base cost for via transitions
            allowed_via_spans: Set of (from_layer, to_layer) pairs allowed for vias.
                              If None, all layer pairs are allowed (full blind/buried support).
                              Layers are indexed 0..N-1.
            use_gpu: Enable GPU acceleration
        """
        # Lateral routing layers: inner layers only (outer layers are reserved
        # for pads/escapes), EXCEPT on 2-layer boards where no inner layers
        # exist and F.Cu/B.Cu must carry the lateral routing themselves.
        if self.layers > 2:
            lateral_layers = range(1, self.layers - 1)
        else:
            lateral_layers = range(self.layers)

        # Count edges to pre-allocate array (avoids MemoryError with 30M edges)
        edge_count = 0

        # Count preferred and optional wrong-way planar edges.
        for z in lateral_layers:
            axes = self.get_allowed_axes(z)
            if "h" in axes:
                edge_count += 2 * self.y_steps * (self.x_steps - 1)
            if "v" in axes:
                edge_count += 2 * self.x_steps * (self.y_steps - 1)

        # Explicit spans win; canonicalize them because add_edge emits both
        # directions for each physical pair.
        if allowed_via_spans is not None:
            legal_via_pairs_set = {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in allowed_via_spans
                if a != b and 0 <= a < self.layers and 0 <= b < self.layers
            }
        else:
            legal_via_pairs_set = self.get_legal_via_pairs(
                self.layers,
                allow_any_layer_via=allow_any_layer_via,
            )
        adjacent_only = (
            self.layers > 2
            and legal_via_pairs_set
            and all(abs(b - a) == 1 for a, b in legal_via_pairs_set)
        )
        # A via is a physical hole and annulus, not a zero-area graph edge.
        # If adjacent lattice sites cannot satisfy the board's center-to-
        # center spacing, restrict vias to a DRC-spaced sublattice. A
        # checkerboard preserves half the sites while increasing the nearest
        # site distance from one pitch to sqrt(2) pitches.
        default_spacing = max(0.0, float(min_via_center_spacing))
        explicit_spacing = {
            canonical_pair(int(a), int(b)): max(0.0, float(spacing))
            for (a, b), spacing in (via_pair_center_spacing or {}).items()
        }
        pair_spacing = {
            pair: explicit_spacing.get(pair, default_spacing)
            for pair in legal_via_pairs_set
        }
        pair_policies = {
            pair: self._via_site_policy(spacing, self.pitch)
            for pair, spacing in pair_spacing.items()
        }
        self._via_pair_site_policies = pair_policies

        # Calls without a layer pair use the most restrictive process. This
        # preserves the historical conservative answer for diagnostics while
        # graph construction uses the exact process for each span.
        required_spacing = max(pair_spacing.values(), default=default_spacing)
        via_site_mode, via_site_stride = self._via_site_policy(
            required_spacing, self.pitch
        )
        self._via_site_mode = via_site_mode
        self._via_site_stride = via_site_stride

        pair_site_counts = {
            pair: sum(
                self.is_via_site(x, y, *pair)
                for x in range(self.x_steps)
                for y in range(self.y_steps)
            )
            for pair in legal_via_pairs_set
        }
        via_site_count = min(
            pair_site_counts.values(), default=0
        )
        via_edge_count = 2 * sum(pair_site_counts.values())
        edge_count += via_edge_count

        logger.info(
            "Pre-allocating for %s edges (%s via edges for %d pairs; "
            "most restrictive process uses %s/%s sites at %.4fmm)",
            f"{edge_count:,}",
            f"{via_edge_count:,}",
            len(legal_via_pairs_set),
            f"{via_site_count:,}",
            f"{self.x_steps * self.y_steps:,}",
            required_spacing,
        )
        graph = CSRGraph(use_gpu, edge_capacity=edge_count)

        # Build lateral edges (H/V discipline)
        for z in lateral_layers:
            direction = self.layer_dir[z]

            if direction == 'h':
                for y in range(self.y_steps):
                    for x in range(self.x_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x+1, y, z)

                        # MANHATTAN VALIDATION
                        if not self.is_legal_planar_edge(x, y, z, x+1, y, z):
                            logger.error(f"[MANHATTAN-VIOLATION] Illegal H edge on layer {z}: ({x},{y}) → ({x+1},{y})")
                            continue  # Skip illegal edge

                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)
            else:  # direction == 'v'
                for x in range(self.x_steps):
                    for y in range(self.y_steps - 1):
                        u = self.node_idx(x, y, z)
                        v = self.node_idx(x, y+1, z)

                        # MANHATTAN VALIDATION
                        if not self.is_legal_planar_edge(x, y, z, x, y+1, z):
                            logger.error(f"[MANHATTAN-VIOLATION] Illegal V edge on layer {z}: ({x},{y}) → ({x},{y+1})")
                            continue  # Skip illegal edge

                        graph.add_edge(u, v, self.pitch)
                        graph.add_edge(v, u, self.pitch)

        # Add the nonpreferred axis when guided or bidirectional routing is
        # enabled. Preferred edges above retain their historical base cost.
        if np.isfinite(self.wrong_way_cost_multiplier):
            wrong_way_cost = (
                self.pitch * self.wrong_way_cost_multiplier
            )
            for z in lateral_layers:
                preferred = self.get_legal_axis(z)
                if preferred == "h":
                    segments = (
                        (x, y, x, y + 1)
                        for x in range(self.x_steps)
                        for y in range(self.y_steps - 1)
                    )
                else:
                    segments = (
                        (x, y, x + 1, y)
                        for y in range(self.y_steps)
                        for x in range(self.x_steps - 1)
                    )
                for x0, y0, x1, y1 in segments:
                    u = self.node_idx(x0, y0, z)
                    v = self.node_idx(x1, y1, z)
                    graph.add_edge(u, v, wrong_way_cost)
                    graph.add_edge(v, u, wrong_way_cost)

        # Build via edges using the SAME legal pairs as pre-allocation
        via_count = 0

        for x in range(self.x_steps):
            for y in range(self.y_steps):
                for (z_from, z_to) in legal_via_pairs_set:
                    if not self.is_via_site(
                        x, y, z_from, z_to
                    ):
                        continue
                    # Only add if this specific pair is legal
                    span = abs(z_to - z_from)
                    if adjacent_only:
                        cost = via_cost * adjacent_via_step_scale
                    else:
                        span_alpha = 0.15
                        cost = via_cost * (1.0 + span_alpha * (span - 1))

                    u = self.node_idx(x, y, z_from)
                    v = self.node_idx(x, y, z_to)
                    graph.add_edge(u, v, cost)
                    graph.add_edge(v, u, cost)
                    via_count += 2

        # LOG what was built
        logger.info(f"Vias: {via_count} edges created")
        logger.info(
            "[VIA-SITES] mode=%s stride=%d sites=%d/%d "
            "nearest_required=%.4fmm",
            via_site_mode,
            via_site_stride,
            via_site_count,
            self.x_steps * self.y_steps,
            required_spacing,
        )
        logger.info(f"Via policy: {len(legal_via_pairs_set)} layer pairs (FULL BLIND/BURIED ENABLED!)")
        for pair in sorted(list(legal_via_pairs_set))[:10]:
            logger.info(f"  Legal via: {pair[0]} ↔ {pair[1]}")
        if len(legal_via_pairs_set) > 20:
            logger.info(f"  ... and {len(legal_via_pairs_set) - 10} more pairs (showing first 10 only)")

        # Finalize the graph before validation (converts edge list to CSR format)
        graph.finalize(self.num_nodes, num_layers=self.layers)

        # MANHATTAN VALIDATION: Sample 1000 random edges and verify they're legal
        edge_count = len(graph.indices) if hasattr(graph, 'indices') else 0
        sample_size = min(1000, edge_count)

        if sample_size > 0:
            logger.info(f"[MANHATTAN-VALIDATION] Sampling {sample_size} edges to verify H/V discipline...")
            violations = 0

            # Convert indptr to CPU for validation (if it's on GPU)
            indptr_cpu = graph.indptr if isinstance(graph.indptr, np.ndarray) else graph.indptr.get()

            for _ in range(sample_size):
                # Pick random edge from CSR structure
                edge_idx = random.randint(0, edge_count - 1)

                # Get source node (find which node this edge belongs to) using binary search
                # indptr[u] <= edge_idx < indptr[u+1], so searchsorted gives us u+1
                u = int(np.searchsorted(indptr_cpu, edge_idx, side='right')) - 1

                # Get target node
                v = int(graph.indices[edge_idx]) if isinstance(graph.indices[edge_idx], (int, np.integer)) else int(graph.indices[edge_idx].get())

                # Convert to coordinates
                x_u, y_u, z_u = self.idx_to_coord(u)
                x_v, y_v, z_v = self.idx_to_coord(v)

                # Convert to Python ints for set membership testing
                z_u, z_v = int(z_u), int(z_v)

                # Check if it's a via (different layers)
                if z_u != z_v:
                    # Via edge - check if it's in legal pairs
                    if (z_u, z_v) not in legal_via_pairs_set and (z_v, z_u) not in legal_via_pairs_set:
                        logger.error(f"[MANHATTAN-VIOLATION] Illegal via: layer {z_u} ↔ {z_v} at ({x_u},{y_u})")
                        violations += 1
                else:
                    # Planar edge - check H/V discipline
                    if not self.is_legal_planar_edge(x_u, y_u, z_u, x_v, y_v, z_v):
                        logger.error(f"[MANHATTAN-VIOLATION] Illegal planar edge on layer {z_u}: ({x_u},{y_u}) → ({x_v},{y_v})")
                        violations += 1

            if violations > 0:
                logger.error(f"[MANHATTAN-VALIDATION] Found {violations} illegal edges in graph!")
                raise RuntimeError("Graph contains non-Manhattan edges")
            else:
                logger.info(f"[MANHATTAN-VALIDATION] All {sample_size} sampled edges are legal ✓")

        return graph


# ═══════════════════════════════════════════════════════════════════════════════
# ROI EXTRACTION (GPU-Accelerated BFS)
# ═══════════════════════════════════════════════════════════════════════════════

class ROIExtractor:
    """Extract Region of Interest subgraph using GPU-vectorized BFS"""

    def __init__(self, graph: CSRGraph, use_gpu: bool = False, lattice=None):
        self.graph = graph
        self.xp = graph.xp
        self.N = len(graph.indptr) - 1
        self.lattice = lattice  # Need lattice for geometric ROI

    def extract_roi_geometric(self, src: int, dst: int, corridor_buffer: int = 30, layer_margin: int = 3, portal_seeds: list = None) -> tuple:
        """
        Geometric bounding box ROI extraction for long nets.
        Much more efficient than BFS for point-to-point routing over long distances.

        Strategy:
        1. Calculate 3D bounding box between src and dst
        2. Add corridor buffer perpendicular to the main routing direction
        3. Limit vertical layers to ±layer_margin from entry/exit layers

        Args:
            src: Source node index
            dst: Destination node index
            corridor_buffer: Perpendicular buffer in grid steps (default: 30 steps = 12mm @ 0.4mm pitch)
            layer_margin: Vertical layer margin from entry/exit layers (default: ±3 layers)

        Returns: (roi_nodes, global_to_roi)
        """
        import numpy as np

        if not self.lattice:
            # Fallback to BFS if no lattice available
            logger.warning("Geometric ROI requires lattice, falling back to BFS")
            return self.extract_roi_bfs(src, dst, initial_radius=40)

        # Get 3D coordinates
        src_x, src_y, src_z = self.lattice.idx_to_coord(src)
        dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst)

        # Calculate axis-aligned bounding box
        min_x = min(src_x, dst_x)
        max_x = max(src_x, dst_x)
        min_y = min(src_y, dst_y)
        max_y = max(src_y, dst_y)
        min_z = min(src_z, dst_z)
        max_z = max(src_z, dst_z)

        # Include portal seeds in bounding box if provided
        if portal_seeds:
            seed_layers = [self.lattice.idx_to_coord(n)[2] for (n, _) in portal_seeds]
            if seed_layers:
                min_z = min(min_z, min(seed_layers))
                max_z = max(max_z, max(seed_layers))

        # Add corridor buffer perpendicular to main direction
        min_x = int(max(0, min_x - corridor_buffer))
        max_x = int(min(self.lattice.x_steps - 1, max_x + corridor_buffer))
        min_y = int(max(0, min_y - corridor_buffer))
        max_y = int(min(self.lattice.y_steps - 1, max_y + corridor_buffer))

        # Add layer margin (clamp to inner layers only - exclude outer layers 0 and layers-1)
        min_z = max(1, min_z - layer_margin)
        max_z = min(self.lattice.layers - 2, max_z + layer_margin)

        # Generate L-shaped corridor using SET for O(1) deduplication
        x_steps = self.lattice.x_steps
        y_steps = self.lattice.y_steps
        plane_size = x_steps * y_steps

        roi_nodes_set = set()  # Use set for O(1) lookups

        # SYMMETRIC L-CORRIDOR: Include BOTH possible L-paths to avoid directional bias

        # L-Path 1: Horizontal first, then vertical (src → (dst.x, src.y) → dst)
        # Horizontal segment: from src.x to dst.x (at src.y with buffer)
        horiz1_min_y = int(max(0, src_y - corridor_buffer))
        horiz1_max_y = int(min(y_steps - 1, src_y + corridor_buffer))

        for z in range(min_z, max_z + 1):
            for y in range(horiz1_min_y, horiz1_max_y + 1):
                for x in range(min_x, max_x + 1):
                    node_idx = z * plane_size + y * x_steps + x
                    roi_nodes_set.add(node_idx)

        # Vertical segment: from src.y to dst.y (at dst.x with buffer)
        vert1_min_x = int(max(0, dst_x - corridor_buffer))
        vert1_max_x = int(min(x_steps - 1, dst_x + corridor_buffer))

        for z in range(min_z, max_z + 1):
            for y in range(min_y, max_y + 1):
                for x in range(vert1_min_x, vert1_max_x + 1):
                    node_idx = z * plane_size + y * x_steps + x
                    roi_nodes_set.add(node_idx)

        # L-Path 2: Vertical first, then horizontal (src → (src.x, dst.y) → dst)
        # Vertical segment: from src.y to dst.y (at src.x with buffer)
        vert2_min_x = int(max(0, src_x - corridor_buffer))
        vert2_max_x = int(min(x_steps - 1, src_x + corridor_buffer))

        for z in range(min_z, max_z + 1):
            for y in range(min_y, max_y + 1):
                for x in range(vert2_min_x, vert2_max_x + 1):
                    node_idx = z * plane_size + y * x_steps + x
                    roi_nodes_set.add(node_idx)

        # Horizontal segment: from src.x to dst.x (at dst.y with buffer)
        horiz2_min_y = int(max(0, dst_y - corridor_buffer))
        horiz2_max_y = int(min(y_steps - 1, dst_y + corridor_buffer))

        for z in range(min_z, max_z + 1):
            for y in range(horiz2_min_y, horiz2_max_y + 1):
                for x in range(min_x, max_x + 1):
                    node_idx = z * plane_size + y * x_steps + x
                    roi_nodes_set.add(node_idx)

        roi_nodes = np.array(list(roi_nodes_set), dtype=np.int32)
        logger.info(f"Symmetric L-corridor ROI: {len(roi_nodes):,} nodes (both L-paths included), Z {min_z}-{max_z}")

        # CRITICAL: Ensure src, dst, AND all portal seeds are in ROI BEFORE truncation
        must_keep_nodes = [src, dst]
        if portal_seeds:
            # Add all portal seed node IDs to must-keep list
            for node_id, _ in portal_seeds:
                if node_id not in must_keep_nodes:
                    must_keep_nodes.append(node_id)

        # Add any missing must-keep nodes
        for node in must_keep_nodes:
            if node not in roi_nodes_set:
                roi_nodes_set.add(node)

        roi_nodes = np.array(list(roi_nodes_set), dtype=np.int32)

        # Cap ROI size if enormous (500K is ~96% of graph, allows access to empty channels)
        max_nodes = getattr(self, "max_roi_nodes", 500_000)  # Balanced: prevents worst hangs but allows long-distance routing
        if roi_nodes.size > max_nodes:
            logger.warning(f"Geometric ROI {roi_nodes.size:,} > {max_nodes:,}, truncating to {max_nodes} (keeping {len(must_keep_nodes)} critical nodes)")

            # CONNECTIVITY-PRESERVING TRUNCATION: BFS growth from src/dst
            # This ensures src's neighbors are included so GPU wavefront can expand
            from collections import deque

            # Build candidate set and neighbor lookup
            cand_set = set(roi_nodes.tolist())

            def get_neighbors(node_idx):
                """Get 4-way (+ via) neighbors for this node"""
                neighbors = []
                x_steps = self.lattice.x_steps
                y_steps = self.lattice.y_steps
                plane_size = x_steps * y_steps

                z, r = divmod(node_idx, plane_size)
                y, x = divmod(r, x_steps)

                # 4-way horizontal
                if x > 0:
                    neighbors.append(node_idx - 1)
                if x + 1 < x_steps:
                    neighbors.append(node_idx + 1)
                if y > 0:
                    neighbors.append(node_idx - x_steps)
                if y + 1 < y_steps:
                    neighbors.append(node_idx + x_steps)

                # Vias (vertical)
                if z > 0:
                    neighbors.append(node_idx - plane_size)
                if z + 1 < self.lattice.layers:
                    neighbors.append(node_idx + plane_size)

                return neighbors

            # BFS from src and dst alternating (to keep both regions connected)
            selected = set(must_keep_nodes)
            q_src = deque([src])
            q_dst = deque([dst])

            # Add portal seeds to their appropriate queues
            if portal_seeds:
                for node_id, _ in portal_seeds:
                    if node_id != src and node_id != dst:
                        # Add to src queue (arbitrary choice)
                        q_src.append(node_id)

            toggle = 0
            while len(selected) < max_nodes and (q_src or q_dst):
                # Alternate between src and dst fronts for balanced growth
                q = q_src if toggle == 0 else q_dst
                toggle ^= 1

                if not q:
                    continue

                # Process one node from this frontier
                u = q.popleft()
                for v in get_neighbors(u):
                    if v in selected:
                        continue
                    if v not in cand_set:
                        continue  # Must be in geometric corridor

                    selected.add(v)
                    q.append(v)

                    if len(selected) >= max_nodes:
                        break

                if len(selected) >= max_nodes:
                    break

            roi_nodes = np.array(list(selected), dtype=np.int32)
            logger.info(f"After BFS truncation: {len(roi_nodes)} nodes (connectivity-preserving from src/dst) vs {max_nodes} budget")

            # DEBUG: Verify src's immediate neighbors are included
            # DISABLED: This BFS reachability check is too expensive (4s per net)
            # and causes test timeouts. The wavefront will naturally fail if ROI is disconnected.
            #
            # src_neighbors = set(get_neighbors(src))
            # neighbors_in_roi = src_neighbors & selected
            # logger.info(f"[BFS-DEBUG] Src {src} has {len(src_neighbors)} neighbors, {len(neighbors_in_roi)} are in BFS-selected ROI")
            #
            # # ROI REACHABILITY CHECK: Verify dst is reachable from src within truncated ROI
            # logger.info(f"[ROI-REACHABILITY] Testing if dst {dst} is reachable from src {src} within truncated ROI...")
            #
            # # Quick BFS to check connectivity (ignoring costs)
            # from collections import deque
            # queue = deque([src])
            # visited_bfs = {src}
            # found_dst = False
            # hop_count = 0
            # max_hops = 10000  # Safety limit
            #
            # while queue and hop_count < max_hops:
            #     u = queue.popleft()
            #     u = int(u)  # CRITICAL: Cast to Python int
            #     hop_count += 1
            #
            #     if u == dst:
            #         found_dst = True
            #         logger.info(f"[ROI-REACHABILITY] ✓ dst {dst} REACHABLE from src {src} in {hop_count} hops")
            #         break
            #
            #     # Get neighbors from graph - CAST TO INT
            #     u_start = int(self.graph.indptr[u])
            #     u_end = int(self.graph.indptr[u + 1])
            #
            #     for e in range(u_start, u_end):
            #         v = int(self.graph.indices[e])  # CAST neighbor to int
            #
            #         # Only expand within ROI
            #         if v not in selected:
            #             continue
            #
            #         if v not in visited_bfs:
            #             visited_bfs.add(v)
            #             queue.append(v)
            #
            # if not found_dst:
            #     logger.error(f"[ROI-REACHABILITY] ✗ dst {dst} NOT REACHABLE from src {src} within truncated ROI!")
            #     logger.error(f"[ROI-REACHABILITY] ROI has {len(roi_nodes)} nodes but src/dst are DISCONNECTED")
            #     logger.error(f"[ROI-REACHABILITY] Need to expand ROI budget or fix truncation logic")

        # Build global_to_roi mapping
        global_to_roi = np.full(self.N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)

        # Log statistics
        x_span = max_x - min_x + 1
        y_span = max_y - min_y + 1
        z_span = max_z - min_z + 1
        logger.debug(f"Geometric ROI: {len(roi_nodes):,} nodes ({x_span}×{y_span}×{z_span} box, buffer={corridor_buffer}, layer_margin={layer_margin})")

        return roi_nodes, global_to_roi

    def extract_roi(self, src: int, dst: int, initial_radius: int = 40, stagnation_bonus: float = 0.0, portal_seeds: list = None) -> tuple:
        """BFS ROI extraction with adaptive radius"""
        import numpy as np

        # Calculate radius based on Manhattan distance (70% from each end = 140% coverage)
        if self.lattice:
            src_x, src_y, src_z = self.lattice.idx_to_coord(src)
            dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst)
            manhattan_dist = abs(dst_x - src_x) + abs(dst_y - src_y)
            # Ensure radius covers full path length for wavefront to succeed
            radius = max(60, int(manhattan_dist * 0.75 + stagnation_bonus * 2.0))
            logger.debug(f"BFS ROI: dist={manhattan_dist}, radius={radius}")
        else:
            radius = initial_radius

        return self.extract_roi_bfs(src, dst, initial_radius=radius, stagnation_bonus=stagnation_bonus, portal_seeds=portal_seeds)

    def extract_roi_bfs(self, src: int, dst: int, initial_radius: int = 40, stagnation_bonus: float = 0.0, portal_seeds: list = None) -> tuple:
        """
        Bidirectional BFS ROI extraction - expands until both src and dst are covered.
        Good for short/medium nets with complex obstacle navigation.
        Returns: (roi_nodes, global_to_roi)
        """
        import numpy as np
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        # Hard-blocked edges (keepouts) must be INVISIBLE to ROI growth: the
        # bidirectional BFS stops when the waves meet, and a cost-blind BFS
        # meets straight through a keepout - the resulting corridor then
        # contains no legal detour and Dijkstra is forced to pay the block
        # cost. Skipping blocked edges makes the waves grow around the
        # keepout so the ROI includes a legal route. The mask is precomputed
        # by _apply_keepout_obstacles (None when the board has no keepouts).
        blocked = getattr(self.graph, 'blocked_edges', None)

        N = self.N
        seen = np.zeros(N, dtype=np.uint8)   # 0=unseen, 1=src-wave, 2=dst-wave, 3=both
        q_src = [src]
        q_dst = [dst]
        seen[src] = 1
        seen[dst] = 2

        depth = 0
        # Apply stagnation bonus: +0.6mm per stagnation mark (grid_pitch=0.4mm → ~1.5 steps)
        # CRITICAL: Limit max_depth to prevent full-board expansion
        # With radius=60, max_depth=80 gives ~32mm radius (covers most routes)
        # Board is 244×227mm, so depth=800 would cover ENTIRE board!
        max_depth = min(int(initial_radius + stagnation_bonus * 2.0), 80)
        met = False
        meeting_depth = None
        # Stopping as soon as the two waves first touch produces a
        # shortest-hop corridor, but negotiated congestion needs several
        # alternate lanes around that corridor.  Keep a bounded halo beyond
        # the first meeting instead of expanding all the way to max_depth.
        # Grow the halo slightly after stagnation so a squeezed route gains
        # alternatives without making every initial ROI board-sized.
        post_meet_halo = min(
            16,
            2 + int(max(0.0, stagnation_bonus) / 0.6),
        )

        # Limit ROI size for efficiency - smaller ROIs converge MUCH faster on GPU!
        # 50K nodes = ~60×60×12 region (24mm × 24mm × 12 layers @ 0.4mm pitch)
        # This is large enough for most routes but small enough for fast wavefront expansion
        # Target: <50 iterations instead of 500+ on full graph
        max_nodes = getattr(self, "max_roi_nodes", 50_000)

        while depth < max_depth and (q_src or q_dst):
            def step(queue, mark):
                next_q = []
                met_flag = False
                for u in queue:
                    s, e = int(indptr[u]), int(indptr[u+1])
                    for ei in range(s, e):
                        if blocked is not None and blocked[ei]:
                            continue
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
            q_dst, met_dst = step(q_dst, 2)
            if met_dst:
                met = True
            depth += 1

            if met and meeting_depth is None:
                meeting_depth = depth
            if (
                meeting_depth is not None
                and depth >= meeting_depth + post_meet_halo
            ):
                break

            # Early stop if ROI exceeds max size (will be truncated anyway)
            if (seen > 0).sum() > max_nodes * 1.5:
                logger.debug(f"BFS early stop at depth {depth}: ROI size {(seen > 0).sum():,} exceeds {max_nodes * 1.5:,.0f}")
                break

        roi_mask = seen > 0
        roi_nodes = np.where(roi_mask)[0]

        # CRITICAL: Ensure src, dst, AND all portal seeds are in ROI BEFORE truncation
        must_keep_nodes = [src, dst]
        if portal_seeds:
            for node_id, _ in portal_seeds:
                if node_id not in must_keep_nodes:
                    must_keep_nodes.append(node_id)

        # Add any missing must-keep nodes
        missing_nodes = [n for n in must_keep_nodes if n not in roi_nodes]
        if missing_nodes:
            roi_nodes = np.append(roi_nodes, missing_nodes)

        # Truncate if needed (max_nodes defined above)
        if roi_nodes.size > max_nodes:
            logger.warning(f"BFS ROI {roi_nodes.size:,} > {max_nodes:,}, truncating (preserving {len(must_keep_nodes)} critical nodes)")

            # Move ALL must-keep nodes to beginning
            kept_indices = []
            for i, node in enumerate(roi_nodes):
                if node in must_keep_nodes:
                    kept_indices.append(i)

            # Swap to front
            for swap_pos, kept_idx in enumerate(kept_indices):
                if kept_idx >= max_nodes:
                    roi_nodes[kept_idx], roi_nodes[swap_pos] = roi_nodes[swap_pos], roi_nodes[kept_idx]

            roi_nodes = roi_nodes[:max_nodes]
            logger.info(f"Verified {sum(1 for n in must_keep_nodes if n in roi_nodes)}/{len(must_keep_nodes)} critical nodes preserved")

        global_to_roi = np.full(N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)

        return roi_nodes, global_to_roi

    def _check_roi_connectivity(self, src: int, dst: int, roi_nodes, roi_indptr, roi_indices) -> bool:
        """Fast BFS to check if src can reach dst in ROI (~1ms check vs 50-100ms failed GPU routing)."""
        from collections import deque
        import numpy as np

        # Convert to set for O(1) lookup
        if isinstance(roi_nodes, np.ndarray):
            roi_set = set(roi_nodes.tolist() if hasattr(roi_nodes, 'tolist') else roi_nodes)
        elif hasattr(roi_nodes, 'get'):  # CuPy
            roi_set = set(roi_nodes.get().tolist())
        else:
            roi_set = set(roi_nodes) if not isinstance(roi_nodes, set) else roi_nodes

        if src not in roi_set or dst not in roi_set:
            return False

        # Convert to NumPy if CuPy
        if hasattr(roi_indptr, 'get'):
            roi_indptr = roi_indptr.get()
        if hasattr(roi_indices, 'get'):
            roi_indices = roi_indices.get()

        # BFS from src to dst
        visited = {src}
        queue = deque([src])
        nodes_explored = 0

        while queue:
            u = queue.popleft()
            nodes_explored += 1

            if u == dst:
                return True

            # Early termination
            if nodes_explored > len(roi_set) * 0.5:
                break

            for ei in range(int(roi_indptr[u]), int(roi_indptr[u + 1])):
                v = int(roi_indices[ei])
                if v in roi_set and v not in visited:
                    visited.add(v)
                    queue.append(v)

        return False


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
        # Initialize path counters
        self._gpu_path_count = 0
        self._cpu_path_count = 0

    def find_path_roi(self, src: int, dst: int, costs, roi_nodes, global_to_roi,
                      node_penalty=None) -> Optional[List[int]]:
        """Find shortest path within ROI subgraph using heap-based Dijkstra (O(E log V)).

        node_penalty: optional float32 array, ROI-local (len == len(roi_nodes)).
        Added to the edge cost when ENTERING that ROI node - used to price
        nodes owned by other nets (ownership-as-cost) without removing them.
        """
        import numpy as np
        import heapq

        # Use GPU if ROI is large enough and GPU solver available
        roi_size = len(roi_nodes) if hasattr(roi_nodes, '__len__') else roi_nodes.shape[0]
        gpu_threshold = getattr(getattr(self, 'config', None), 'gpu_roi_min_nodes', 1000)
        use_gpu = hasattr(self, 'gpu_solver') and self.gpu_solver and roi_size > gpu_threshold

        if use_gpu:
            logger.info(f"[GPU] Using GPU pathfinding for ROI size={roi_size} (threshold={gpu_threshold})")
            try:
                path = self.gpu_solver.find_path_roi_gpu(
                    src, dst, costs, roi_nodes, global_to_roi,
                    node_penalty=node_penalty,
                )
                if path:
                    self._gpu_path_count += 1
                    return path
                # If GPU returned None, fall through to CPU
            except Exception as e:
                logger.warning(f"[GPU] Pathfinding failed: {e}, using CPU")
                # Fall through to CPU

        # Track CPU pathfinding usage
        self._cpu_path_count += 1

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
                if node_penalty is not None:
                    alt += float(node_penalty[v_roi])
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
                                        costs, roi_nodes, global_to_roi,
                                        node_penalty=None) -> Optional[Tuple[List[int], int, int]]:
        """
        Find shortest path from any source to any destination with portal entry costs.

        node_penalty: optional ROI-local float32 array added when entering a
        node (ownership-as-cost, see find_path_roi).

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
                source_cost = float(initial_cost)
                if node_penalty is not None:
                    source_cost += float(node_penalty[roi_idx])
                dist[roi_idx] = source_cost
                heapq.heappush(heap, (source_cost, roi_idx))
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
                if node_penalty is not None:
                    alt += float(node_penalty[v_roi])
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

        # Environment variable overrides
        env_use_gpu = os.getenv("USE_GPU")
        if env_use_gpu is not None:
            self.config.use_gpu = env_use_gpu == "1"

        env_sequential = os.getenv("SEQUENTIAL_ALL")
        if env_sequential is not None:
            if hasattr(self.config, 'use_gpu_sequential'):
                self.config.use_gpu_sequential = env_sequential == "1"
            else:
                setattr(self.config, 'use_gpu_sequential', env_sequential == "1")

        env_incremental = os.getenv("INCREMENTAL_COST_UPDATE")
        if env_incremental is not None:
            if hasattr(self.config, 'use_incremental_cost_update'):
                self.config.use_incremental_cost_update = env_incremental == "1"
            else:
                setattr(self.config, 'use_incremental_cost_update', env_incremental == "1")

        # Log final configuration
        logger.info(f"[CONFIG] use_gpu={self.config.use_gpu}")
        logger.info(f"[CONFIG] use_gpu_sequential={getattr(self.config, 'use_gpu_sequential', True)}")
        logger.info(f"[CONFIG] use_incremental_cost_update={getattr(self.config, 'use_incremental_cost_update', False)}")

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
        self.portal_candidates: Dict[str, List[Portal]] = {}
        self.net_selected_portals: Dict[
            str, Tuple[Portal, Portal]
        ] = {}
        self.net_portal_failures: Dict[str, int] = defaultdict(int)  # net_id -> failure count
        self.net_pad_ids: Dict[str, Tuple[str, str]] = {}  # net_id -> (src_pad_id, dst_pad_id)
        self.net_portal_layers: Dict[str, Tuple[int, int]] = {}  # net_id -> (entry_layer, exit_layer)
        self._escape_records = {}
        self._escape_spatial = defaultdict(set)
        self._escape_reserved_records = {}
        self._escape_reserved_spatial = defaultdict(set)
        self._escape_bucket_mm = 1.0
        self._portal_barrel_history = defaultdict(float)
        self._portal_cleanup_move_counts = defaultdict(int)

        # Pad escape planner (initialized after lattice is created)
        self.escape_planner: Optional[PadEscapePlanner] = None

        # ROI policy: track stagnation and fallback usage
        self.stagnation_counter: int = 0  # increments each stagnation event
        self.full_graph_fallback_count: int = 0  # limit to 5 per iteration

        # Rip tracking and pres_fac freezing (Fix 5)
        self._last_ripped: Set[str] = set()
        self._last_stagnation_victims: Tuple[str, ...] = ()
        self._stagnation_victim_history: Set[str] = set()
        self._freeze_pres_fac_until: int = 0

        # Connectivity check cache (optimization to avoid redundant BFS checks)
        self._connectivity_cache: Dict[Tuple[int, int, int], bool] = {}  # (src, dst, roi_hash) -> is_connected
        self._connectivity_stats = {
            'checks_performed': 0,
            'cache_hits': 0,
            'disconnected_found': 0,
            'time_saved_ms': 0.0
        }

        # GPU vs CPU pathfinding usage tracking
        self._gpu_path_count = 0  # Number of paths found using GPU
        self._cpu_path_count = 0  # Number of paths found using CPU

        # Legacy attributes for compatibility
        self._instance_tag = f"PF-{int(time.time() * 1000) % 100000}"

        logger.info(f"PathFinder (GPU={self.config.use_gpu and GPU_AVAILABLE}, Portals={self.config.portal_enabled})")

    def initialize_graph(self, board: Board) -> bool:
        """Build routing graph"""
        design_rules = getattr(board, "_design_rules", None) or {}
        hdi_stack = getattr(self.config, "hdi_stack", None)
        if hdi_stack is not None:
            design_rules = dict(design_rules)
            design_rules.update(hdi_stack.design_rules())
            board._design_rules = design_rules
        if design_rules:
            self.config.track_width = float(
                design_rules.get(
                    "default_track_width", self.config.track_width
                )
            )
            self.config.clearance = float(
                design_rules.get(
                    "default_clearance", self.config.clearance
                )
            )
            self.config.via_diameter = float(
                design_rules.get(
                    "default_via_diameter", self.config.via_diameter
                )
            )
            self.config.via_drill = float(
                design_rules.get(
                    "default_via_drill", self.config.via_drill
                )
            )
            self.config.min_hole_to_hole = float(
                design_rules.get(
                    "min_hole_to_hole",
                    self.config.min_hole_to_hole,
                )
            )
            self.config.hole_clearance = float(
                design_rules.get(
                    "min_hole_clearance",
                    self.config.hole_clearance,
                )
            )
        logger.info("=" * 80)
        logger.info("PATHFINDER NEGOTIATED CONGESTION ROUTER - RUNTIME CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"[CONFIG] pres_fac_init    = {self.config.pres_fac_init}")
        logger.info(f"[CONFIG] pres_fac_mult    = {self.config.pres_fac_mult}")
        logger.info(f"[CONFIG] pres_fac_max     = {self.config.pres_fac_max}")
        logger.info(f"[CONFIG] hist_gain        = {self.config.hist_gain}")
        logger.info(f"[CONFIG] via_cost         = {self.config.via_cost}")
        logger.info(f"[CONFIG] grid_pitch       = {self.config.grid_pitch} mm")
        logger.info(
            "[CONFIG] geometry         = %.4fmm track / %.4fmm "
            "clearance / %.4fmm via / %.4fmm drill",
            self.config.track_width,
            self.config.clearance,
            self.config.via_diameter,
            self.config.via_drill,
        )
        logger.info(f"[CONFIG] max_iterations   = {self.config.max_iterations}")
        logger.info(f"[CONFIG] stagnation_patience = {self.config.stagnation_patience}")
        logger.info("=" * 80)

        bounds = self._calc_bounds(board)

        # Use board's real layer count (critical for dense boards)
        layers_from_board = getattr(board, "layer_count", None) or len(getattr(board, "layers", [])) or self.config.layer_count
        self.config.layer_count = int(layers_from_board)
        if hdi_stack is not None:
            if hdi_stack.layer_count != self.config.layer_count:
                raise ValueError(
                    f"{hdi_stack.name} requires {hdi_stack.layer_count} "
                    f"layers, board has {self.config.layer_count}"
                )
            self.config.allowed_via_spans = set(
                hdi_stack.allowed_via_spans
            )

        # Domain boards store Layer objects, while geometry and keepout code
        # require string names. Prefer the board's exact copper stackup; the
        # config default may have the wrong length or omit B.Cu after slicing.
        board_layer_names = [
            layer.name if hasattr(layer, "name") else str(layer)
            for layer in (getattr(board, "layers", None) or [])
            if (layer.name if hasattr(layer, "name") else str(layer)).endswith(".Cu")
        ]
        existing_names = getattr(self.config, "layer_names", None)
        existing_names = [
            layer.name if hasattr(layer, "name") else str(layer)
            for layer in existing_names
        ] if isinstance(existing_names, (list, tuple)) else []

        if len(board_layer_names) == self.config.layer_count:
            self.config.layer_names = board_layer_names
        elif len(existing_names) == self.config.layer_count:
            self.config.layer_names = existing_names
        else:
            self.config.layer_names = (
                ["F.Cu"]
                + [f"In{i}.Cu" for i in range(1, self.config.layer_count - 1)]
                + (["B.Cu"] if self.config.layer_count > 1 else [])
            )

        logger.info(f"Using {self.config.layer_count} layers from board")
        if hdi_stack is not None:
            logger.info(
                "[HDI-STACK] %s (%s), %d explicit adjacent spans, "
                "central core pair %s",
                hdi_stack.name,
                hdi_stack.notation,
                len(hdi_stack.allowed_via_spans),
                hdi_stack.core_pair,
            )

        preferred_layer_directions = (
            self.config.preferred_layer_directions
        )
        if preferred_layer_directions is None:
            (
                preferred_layer_directions,
                h_layers,
                v_layers,
                demand_h_pct,
            ) = preferred_layer_directions_for_board(
                board,
                self.config.layer_count,
            )
            logger.info(
                "[LAYER-DIRECTIONS] Demand-aware graph assignment: "
                "%d H / %d V internal layers for %.1f%% H demand "
                "(H=%s, V=%s)",
                len(h_layers),
                len(v_layers),
                demand_h_pct * 100.0,
                sorted(h_layers),
                sorted(v_layers),
            )

        self.lattice = Lattice3D(
            bounds,
            self.config.grid_pitch,
            self.config.layer_count,
            preferred_layer_directions=preferred_layer_directions,
            wrong_way_cost_multiplier=getattr(
                self.config,
                "wrong_way_cost_multiplier",
                float("inf"),
            ),
        )

        allow_any_layer_via = getattr(
            self.config, "allow_any_layer_via", None
        )
        if allow_any_layer_via is None:
            allow_any_layer_via = self.config.layer_count <= 18
        logger.info(
            "[VIA-TOPOLOGY] %s spans for %d layers",
            "full" if allow_any_layer_via else "adjacent",
            self.config.layer_count,
        )

        self.graph = self.lattice.build_graph(
            self.config.via_cost,
            allowed_via_spans=self.config.allowed_via_spans,
            use_gpu=self.config.use_gpu and GPU_AVAILABLE,
            allow_any_layer_via=allow_any_layer_via,
            adjacent_via_step_scale=getattr(
                self.config, "adjacent_via_step_scale", 4.0
            ),
            min_via_center_spacing=max(
                float(self.config.via_diameter)
                + float(self.config.clearance),
                float(self.config.via_drill)
                + float(getattr(
                    self.config, "min_hole_to_hole", 0.0
                )),
            ),
            via_pair_center_spacing=(
                hdi_stack.center_spacing_by_span()
                if hdi_stack is not None else None
            ),
        )
        # Lazily populated by _path_to_edges. Invalidate if this router is
        # reinitialized with a different board.
        self._indptr_cpu = None
        self._indices_cpu = None
        self._canonical_edge_resource_mask_cache = None
        # Note: graph.finalize() is now called inside build_graph() before validation

        # Set N for ROI checks (number of nodes in full graph)
        self.N = self.lattice.num_nodes

        E = len(self.graph.indices)
        self.accounting = EdgeAccountant(E, use_gpu=self.config.use_gpu and GPU_AVAILABLE)

        # Via pooling arrays (GPU-accelerated for performance!)
        Nx, Ny, Nz = self.lattice.x_steps, self.lattice.y_steps, self.lattice.layers
        self._Nx, self._Ny, self._Nz = Nx, Ny, Nz

        # Layer balancing (EWMA of per-layer horizontal overuse)
        self.layer_bias = np.ones(Nz, dtype=np.float32)  # Index by z (0..Nz-1), 1.0 = neutral
        logger.info(f"[LAYER-BALANCE] Initialized for {Nz} layers")

        # Determine if we should use GPU for via arrays
        use_via_gpu = self.config.use_gpu and GPU_AVAILABLE

        if getattr(self.config, "via_column_pooling", True):
            # Create arrays on GPU if available, CPU otherwise
            if use_via_gpu:
                self.via_col_cap = cp.full((Nx, Ny), int(getattr(self.config, "via_column_capacity", 4)), dtype=cp.int16)
                self.via_col_use = cp.zeros((Nx, Ny), dtype=cp.int16)
                self.via_col_pres = cp.zeros((Nx, Ny), dtype=cp.float32)
                logger.info(f"[VIA-POOL] Column pooling enabled (GPU): capacity={int(self.via_col_cap[0,0])} per (x,y)")
            else:
                self.via_col_cap = np.full((Nx, Ny), int(getattr(self.config, "via_column_capacity", 4)), dtype=np.int16)
                self.via_col_use = np.zeros((Nx, Ny), dtype=np.int16)
                self.via_col_pres = np.zeros((Nx, Ny), dtype=np.float32)
                logger.info(f"[VIA-POOL] Column pooling enabled (CPU): capacity={self.via_col_cap[0,0]} per (x,y)")

        if getattr(self.config, "via_segment_pooling", True) and Nz > 2:
            # Segments between routing layers (1..Nz-2): segment z→z+1 stored at index z-1
            # Skipped entirely when Nz <= 2: a 2-layer board has only through
            # vias and no segments to pool (consumers all hasattr-guard).
            self._segZ = Nz - 2  # Number of routing layers
            if use_via_gpu:
                self.via_seg_cap = cp.full((Nx, Ny, self._segZ), int(getattr(self.config, "via_segment_capacity", 2)), dtype=cp.int8)
                self.via_seg_use = cp.zeros((Nx, Ny, self._segZ), dtype=cp.int16)
                self.via_seg_pres = cp.zeros((Nx, Ny, self._segZ), dtype=cp.float32)
                self.via_seg_prefix = cp.zeros((Nx, Ny, self._segZ), dtype=cp.float32)
                logger.info(f"[VIA-POOL] Segment pooling enabled (GPU): {self._segZ} segments (z=1..{Nz-2}), capacity={int(self.via_seg_cap[0,0,0])} per segment")
            else:
                self.via_seg_cap = np.full((Nx, Ny, self._segZ), int(getattr(self.config, "via_segment_capacity", 2)), dtype=np.int8)
                self.via_seg_use = np.zeros((Nx, Ny, self._segZ), dtype=np.int16)
                self.via_seg_pres = np.zeros((Nx, Ny, self._segZ), dtype=np.float32)
                self.via_seg_prefix = np.zeros((Nx, Ny, self._segZ), dtype=np.float32)
                logger.info(f"[VIA-POOL] Segment pooling enabled (CPU): {self._segZ} segments (z=1..{Nz-2}), capacity={self.via_seg_cap[0,0,0]} per segment")

        # Initialize ViaKernelManager for GPU-accelerated via operations
        self.via_kernel_manager = ViaKernelManager(use_gpu=use_via_gpu)
        logger.info(f"[VIA-KERNELS] Manager initialized (GPU={'YES' if use_via_gpu else 'NO'})")

        # NODE OWNERSHIP TRACKING: Track which net owns each node (via barrels)
        # -1 = free, otherwise net_id (mapped to integer)
        # This is THE solution to via barrel conflicts - enforce at node level, not edge level!
        self.node_owner = np.full(self.lattice.num_nodes, -1, dtype=np.int32)
        self._node_owner_members: Dict[int, Set[int]] = {}
        self.node_owner_gpu = None
        # Explicit terminal vias are physically off-grid. Their clearance
        # footprints therefore cannot be represented by graph via edges, but
        # later graph searches still need to price tracks and vias entering
        # those footprints.
        self.portal_clearance_owner = np.full(
            self.lattice.num_nodes, -1, dtype=np.int32
        )
        self._portal_clearance_owner_members: Dict[int, Set[int]] = {}
        self.portal_clearance_owner_gpu = None
        self._portal_clearance_nodes_cache = {}
        self._portal_clearance_halo_cache = {}
        self._portal_clearance_xy_cache = {}
        self.path_node_use = np.zeros(
            self.lattice.num_nodes, dtype=np.int16
        )
        self.path_node_use_gpu = None
        self.node_conflict_history = np.zeros(
            self.lattice.num_nodes, dtype=np.float32
        )
        self.node_conflict_history_gpu = None
        self.net_id_map = {}  # net_name -> integer ID
        self.next_net_id = 0
        logger.info(f"[NODE-OWNER] Initialized node ownership tracking for {self.lattice.num_nodes:,} nodes")

        self.solver = SimpleDijkstra(self.graph, self.lattice)

        # Apple Silicon: graft the Metal/MLX solver over the CPU one
        # (ORTHO_BACKEND=metal or config.use_metal; CPU fallback retained)
        if os.getenv("ORTHO_BACKEND") == "metal" or getattr(self.config, "use_metal", False):
            from .pathfinder.metal_dijkstra import try_attach_metal
            try_attach_metal(self.solver, self.graph, self.lattice)

        # Add GPU solver if available
        use_gpu_solver = self.config.use_gpu and GPU_AVAILABLE and CUDA_DIJKSTRA_AVAILABLE

        # Enhanced debug logging
        logger.info(f"[GPU-INIT] config.use_gpu={self.config.use_gpu}, GPU_AVAILABLE={GPU_AVAILABLE}, CUDA_DIJKSTRA_AVAILABLE={CUDA_DIJKSTRA_AVAILABLE}")
        logger.info(f"[GPU-INIT] use_gpu_solver={use_gpu_solver}")

        if use_gpu_solver:
            try:
                self.solver.gpu_solver = CUDADijkstra(self.graph, self.lattice)
                self.node_owner_gpu = cp.asarray(self.node_owner)
                self.portal_clearance_owner_gpu = cp.asarray(
                    self.portal_clearance_owner
                )
                self.path_node_use_gpu = cp.asarray(self.path_node_use)
                self.node_conflict_history_gpu = cp.asarray(
                    self.node_conflict_history
                )
                logger.info("[GPU] CUDA Near-Far Dijkstra enabled (ROI > 5K nodes) with lattice dims")
                # Log GPU details
                device = cp.cuda.Device()
                mem_free, mem_total = device.mem_info
                logger.info(f"[GPU] GPU Compute Capability: {device.compute_capability}")
                logger.info(f"[GPU] GPU Memory: {mem_free / 1e9:.1f} GB free / {mem_total / 1e9:.1f} GB total")
            except Exception as e:
                logger.warning(f"[GPU] Failed to initialize CUDA Dijkstra: {e}")
                self.solver.gpu_solver = None
        else:
            self.solver.gpu_solver = None
            reasons = []
            if not self.config.use_gpu:
                reasons.append("config.use_gpu=False")
            if not GPU_AVAILABLE:
                reasons.append("CuPy not installed")
            if not CUDA_DIJKSTRA_AVAILABLE:
                reasons.append("CUDADijkstra import failed")
            logger.info(f"[GPU] CPU-only mode: {', '.join(reasons)}")
        self.roi_extractor = ROIExtractor(self.graph, use_gpu=self.config.use_gpu and GPU_AVAILABLE, lattice=self.lattice)

        # Identify via edges for via-specific accounting
        self._identify_via_edges()

        # Build via edge metadata for vectorized penalty application
        self._build_via_edge_metadata()

        self._map_pads(board)

        # Initialize escape planner after pads are mapped
        # Use deterministic random seed for reproducible escape planning (default: 42)
        escape_seed = getattr(self.config, 'escape_random_seed', 42)
        self.escape_planner = PadEscapePlanner(self.lattice, self.config, self.pad_to_node, random_seed=escape_seed)

        # NOTE: Portal planning is now done by PadEscapePlanner.precompute_all_pad_escapes()
        # which is called from main_window.py AFTER initialization.
        # The old _plan_portals() method is disabled in favor of the column-based algorithm.
        # Portals will be copied from escape_planner in precompute_all_pad_escapes().
        logger.info(f"Portal planning delegated to PadEscapePlanner (column-based, seed={escape_seed})")

        # Note: Portal discounts are applied at seed level in _get_portal_seeds()
        # No need for graph-level discount modification

        # Block edges inside keepout rule areas (from PR #17 by RolandWa)
        self._apply_keepout_obstacles(board)

        logger.info("=== Init complete ===")
        return True

    def _apply_keepout_obstacles(self, board) -> None:
        """Block lattice edges inside keepout rule area polygons.

        Respects per-keepout constraint flags:
            keepout_tracks -> block planar (same-layer) edges
            keepout_vias   -> block via (inter-layer) edges

        Called from initialize_graph() after the graph is built.
        (From PR #17 by RolandWa, with dst-layer lookup vectorised.)
        """
        keepouts = getattr(board, 'keepouts', [])
        if not keepouts:
            return

        if getattr(self, 'graph', None) is None or self.graph.base_costs is None:
            logger.warning("[KEEPOUT] Graph not initialized, skipping keepout obstacles")
            return

        base_cost = self.graph.base_costs
        is_gpu = hasattr(base_cost, 'get')
        base_cost_cpu = base_cost.get() if is_gpu else base_cost

        indptr = self.graph.indptr
        indices = self.graph.indices
        if hasattr(indptr, 'get'):
            indptr = indptr.get()
        if hasattr(indices, 'get'):
            indices = indices.get()

        BLOCK_COST = 1e9
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
        plane = Nx * Ny
        bounds = self.lattice.bounds
        pitch = self.lattice.pitch

        # Pre-build vectorised mm grids for PIP tests (shape Nx*Ny)
        xs = bounds[0] + np.arange(Nx, dtype=np.float64) * pitch
        ys = bounds[1] + np.arange(Ny, dtype=np.float64) * pitch
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        grid_x_flat = grid_x.ravel()
        grid_y_flat = grid_y.ravel()

        layer_names = list(getattr(self.config, 'layer_names', []))
        total_blocked_tracks = 0
        total_blocked_vias = 0

        for keepout in keepouts:
            outline = keepout.get('outline', [])
            if len(outline) < 3:
                continue

            block_tracks = keepout.get('keepout_tracks', False)
            block_vias = keepout.get('keepout_vias', False)
            if not block_tracks and not block_vias:
                continue

            # Determine affected z-indices (all layers if unmapped)
            ko_layers = keepout.get('layers', [])
            if ko_layers and layer_names:
                z_indices = [layer_names.index(ln) for ln in ko_layers if ln in layer_names]
            else:
                z_indices = list(range(Nz))
            if not z_indices:
                continue

            # Vectorised point-in-polygon over the whole (Nx, Ny) plane
            poly = np.array(outline, dtype=np.float64)
            inside_2d = _points_in_polygon(grid_x_flat, grid_y_flat, poly).reshape(Nx, Ny)
            xis, yis = np.nonzero(inside_2d)
            if len(xis) == 0:
                continue

            for zi in z_indices:
                node_idxs = zi * plane + yis * Nx + xis
                for node_idx in node_idxs.tolist():
                    start, end = int(indptr[node_idx]), int(indptr[node_idx + 1])
                    if start == end:
                        continue
                    dst_z = indices[start:end] // plane
                    is_via = dst_z != zi
                    if block_tracks:
                        planar_eids = np.arange(start, end)[~is_via]
                        base_cost_cpu[planar_eids] = BLOCK_COST
                        total_blocked_tracks += len(planar_eids)
                    if block_vias:
                        via_eids = np.arange(start, end)[is_via]
                        base_cost_cpu[via_eids] = BLOCK_COST
                        total_blocked_vias += len(via_eids)

        # Sync back to GPU if needed
        if is_gpu:
            self.graph.base_costs = cp.asarray(base_cost_cpu)
        else:
            self.graph.base_costs = base_cost_cpu

        # Precomputed mask for the ROI extractor (BFS must not grow through
        # blocked edges - see extract_roi_bfs)
        self.graph.blocked_edges = base_cost_cpu >= 1e8

        logger.info(
            f"[KEEPOUT] Applied {len(keepouts)} keepout area(s): "
            f"blocked {total_blocked_tracks} track edges, {total_blocked_vias} via edges"
        )

    def _calc_bounds(self, board: Board) -> Tuple[float, float, float, float]:
        """
        Compute routing grid bounds from pads extracted via board.nets.

        This is called during initialize_graph() BEFORE escape planning,
        so we extract pads from board.nets (which IS available) rather than
        board.components (which may be incomplete).
        """
        pads_with_nets = []
        ROUTING_MARGIN = 3.0  # mm

        # Extract pads from board.nets (reliable during initialization)
        try:
            if hasattr(board, 'nets') and board.nets:
                for net in board.nets:
                    # Only consider nets with 2+ pads (routable nets)
                    if hasattr(net, 'pads') and len(net.pads) >= 2:
                        for pad in net.pads:
                            if hasattr(pad, 'position') and pad.position is not None:
                                pads_with_nets.append(pad)

                if pads_with_nets:
                    xs = [p.position.x for p in pads_with_nets]
                    ys = [p.position.y for p in pads_with_nets]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    logger.info(f"[BOUNDS] Extracted {len(pads_with_nets)} pads from {len(board.nets)} nets")
                    logger.info(f"[BOUNDS] Pad area: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")

                    # Add routing margin
                    bounds = (
                        min_x - ROUTING_MARGIN,
                        min_y - ROUTING_MARGIN,
                        max_x + ROUTING_MARGIN,
                        max_y + ROUTING_MARGIN
                    )
                    logger.info(f"[BOUNDS] Final with {ROUTING_MARGIN}mm margin: ({bounds[0]:.1f}, {bounds[1]:.1f}) to ({bounds[2]:.1f}, {bounds[3]:.1f})")
                    return bounds

        except Exception as e:
            logger.warning(f"[BOUNDS] Failed to extract pads from board.nets: {e}")

        # Fallback: Use full board bounds + margin (suboptimal but safe)
        logger.warning(f"[BOUNDS] No pads found via board.nets, falling back to board._kicad_bounds + {ROUTING_MARGIN}mm")
        if hasattr(board, "_kicad_bounds"):
            b = board._kicad_bounds
            return (b[0] - ROUTING_MARGIN, b[1] - ROUTING_MARGIN,
                    b[2] + ROUTING_MARGIN, b[3] + ROUTING_MARGIN)

        # Ultimate fallback
        logger.error("[BOUNDS] No bounds available, using default 100x100mm")
        return (0, 0, 100, 100)

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
        """
        Get every legal internal entry layer at a fixed portal cell.

        The escape planner creates an F.Cu stub to this cell.  The winning
        internal seed is connected back to that stub before path commit.
        PathFinder will create the escape via (F.Cu → internal layer) based on routing needs.

        The path cost chooses the winner.  Starting on F.Cu would let a route
        move laterally away from the portal before taking its via.
        """
        seeds = []
        routing_layers = (
            range(1, self.lattice.layers - 1)
            if self.lattice.layers > 2
            else [1]
        )
        if portal.dynamic_entry:
            # The off-grid via reaches its lattice anchor with a short
            # horizontal entry segment, so only horizontal layers are legal.
            routing_layers = [
                layer for layer in routing_layers
                if "h" in self.lattice.get_allowed_axes(layer)
            ]
        via_base = float(getattr(self.config, "via_cost", 0.7))
        discount = float(getattr(
            self.config, "portal_via_discount", 0.15
        ))
        for layer in routing_layers:
            node_idx = self.lattice.node_idx(
                portal.x_idx, portal.y_idx, layer
            )
            depth = abs(layer - portal.pad_layer)
            seeds.append((node_idx, via_base * discount * depth))
        return seeds

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        px, py = point
        ax, ay = start
        bx, by = end
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-24:
            return float(np.hypot(px - ax, py - ay))
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    @staticmethod
    def _segments_intersect(a, b, c, d) -> bool:
        def orient(p, q, r):
            return (
                (q[0] - p[0]) * (r[1] - p[1])
                - (q[1] - p[1]) * (r[0] - p[0])
            )

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) - 1e-12 <= q[0]
                <= max(p[0], r[0]) + 1e-12
                and min(p[1], r[1]) - 1e-12 <= q[1]
                <= max(p[1], r[1]) + 1e-12
            )

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        if (
            ((o1 > 1e-12 and o2 < -1e-12)
             or (o1 < -1e-12 and o2 > 1e-12))
            and ((o3 > 1e-12 and o4 < -1e-12)
                 or (o3 < -1e-12 and o4 > 1e-12))
        ):
            return True
        return (
            (abs(o1) <= 1e-12 and on_segment(a, c, b))
            or (abs(o2) <= 1e-12 and on_segment(a, d, b))
            or (abs(o3) <= 1e-12 and on_segment(c, a, d))
            or (abs(o4) <= 1e-12 and on_segment(c, b, d))
        )

    @classmethod
    def _segment_distance(cls, first, second) -> float:
        a, b = first
        c, d = second
        if cls._segments_intersect(a, b, c, d):
            return 0.0
        return min(
            cls._point_segment_distance(a, c, d),
            cls._point_segment_distance(b, c, d),
            cls._point_segment_distance(c, a, b),
            cls._point_segment_distance(d, a, b),
        )

    def _escape_record(self, net_id: str, pad_id: str, portal: Portal):
        portal_xy = self.escape_planner._portal_world(portal)
        segments = tuple(self.escape_planner._escape_segments(
            portal.pad_x,
            portal.pad_y,
            portal_xy[0],
            portal_xy[1],
        ))
        points = [portal_xy]
        for start, end in segments:
            points.extend((start, end))
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {
            "key": (net_id, pad_id),
            "net": net_id,
            "pad": pad_id,
            "portal": portal,
            "via": portal_xy,
            "segments": segments,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
        }

    def _escape_cells(self, record):
        margin = (
            float(self.config.via_diameter)
            + float(self.config.clearance)
        )
        xmin, ymin, xmax, ymax = record["bbox"]
        bucket = self._escape_bucket_mm
        ix0 = int(np.floor((xmin - margin) / bucket))
        iy0 = int(np.floor((ymin - margin) / bucket))
        ix1 = int(np.floor((xmax + margin) / bucket))
        iy1 = int(np.floor((ymax + margin) / bucket))
        return tuple(
            (ix, iy)
            for ix in range(ix0, ix1 + 1)
            for iy in range(iy0, iy1 + 1)
        )

    def _escape_records_conflict(self, first, second) -> bool:
        clearance = float(self.config.clearance)
        track_width = float(self.config.track_width)
        track_radius = 0.5 * track_width
        via_radius = 0.5 * float(self.config.via_diameter)

        if (
            float(np.hypot(
                first["via"][0] - second["via"][0],
                first["via"][1] - second["via"][1],
            ))
            < 2.0 * via_radius + clearance - 1e-9
        ):
            return True

        via_track_limit = via_radius + track_radius + clearance
        for segment in second["segments"]:
            if (
                self._point_segment_distance(
                    first["via"], segment[0], segment[1]
                )
                < via_track_limit - 1e-9
            ):
                return True
        for segment in first["segments"]:
            if (
                self._point_segment_distance(
                    second["via"], segment[0], segment[1]
                )
                < via_track_limit - 1e-9
            ):
                return True

        track_track_limit = track_width + clearance
        return any(
            self._segment_distance(first_segment, second_segment)
            < track_track_limit - 1e-9
            for first_segment in first["segments"]
            for second_segment in second["segments"]
        )

    def _escape_candidate_conflict_keys(
        self, net_id: str, pad_id: str, portal: Portal,
        records=None, spatial=None,
    ):
        records = self._escape_records if records is None else records
        spatial = self._escape_spatial if spatial is None else spatial
        if not records:
            return set()
        candidate = self._escape_record(net_id, pad_id, portal)
        nearby = set()
        for cell in self._escape_cells(candidate):
            nearby.update(spatial.get(cell, ()))
        return {
            key
            for key in nearby
            if records[key]["net"] != net_id
            and self._escape_records_conflict(
                candidate, records[key]
            )
        }

    def _escape_candidate_conflicts(
        self, net_id: str, pad_id: str, portal: Portal,
        records=None, spatial=None,
    ) -> int:
        return len(self._escape_candidate_conflict_keys(
            net_id,
            pad_id,
            portal,
            records=records,
            spatial=spatial,
        ))

    def _mark_escape_occupancy(
        self, net_id: str, selected_portals: Tuple[Portal, Portal]
    ) -> None:
        pad_ids = self.net_pad_ids.get(net_id)
        if not pad_ids:
            return
        for pad_id, portal in zip(pad_ids, selected_portals):
            if portal is None:
                continue
            record = self._escape_record(net_id, pad_id, portal)
            self._insert_escape_record(record)

    def _insert_escape_record(self, record) -> None:
        cells = self._escape_cells(record)
        record["cells"] = cells
        self._escape_records[record["key"]] = record
        for cell in cells:
            self._escape_spatial[cell].add(record["key"])

    def _remove_escape_record(self, key):
        record = self._escape_records.pop(key, None)
        if record is None:
            return None
        for cell in record["cells"]:
            members = self._escape_spatial.get(cell)
            if members is None:
                continue
            members.discard(key)
            if not members:
                self._escape_spatial.pop(cell, None)
        return record

    def _clear_escape_occupancy(self, net_id: str) -> None:
        keys = [
            key for key in self._escape_records
            if key[0] == net_id
        ]
        for key in keys:
            self._remove_escape_record(key)

    def _rebuild_escape_occupancy(self) -> None:
        self._escape_records.clear()
        self._escape_spatial.clear()
        for net_id, selected in self.net_selected_portals.items():
            if self.net_paths.get(net_id):
                self._mark_escape_occupancy(net_id, selected)

    def _detect_escape_conflicts(self):
        pairs = set()
        for key, record in self._escape_records.items():
            nearby = set()
            for cell in record["cells"]:
                nearby.update(self._escape_spatial.get(cell, ()))
            for other_key in nearby:
                if other_key <= key:
                    continue
                other = self._escape_records[other_key]
                if other["net"] == record["net"]:
                    continue
                if self._escape_records_conflict(record, other):
                    pairs.add((key, other_key))

        owners = {pair[0][0] for pair in pairs}
        victims = {pair[1][0] for pair in pairs}
        return pairs, owners, victims

    def _escape_conflict_portal_keys(self, pairs):
        """Return the selected portal-history keys involved in conflicts."""
        keys = set()
        for pair in pairs:
            for record_key in pair:
                record = self._escape_records[record_key]
                portal = record["portal"]
                keys.add((
                    record["pad"],
                    portal.x_idx,
                    portal.y_idx,
                ))
        return keys

    def _plan_escape_assignment(self) -> None:
        """Choose a globally clearance-aware portal candidate for every pad."""
        entries = []
        for net_id, pad_ids in self.net_pad_ids.items():
            for pad_id in pad_ids:
                candidates = self.portal_candidates.get(pad_id)
                if not candidates:
                    primary = self.portals.get(pad_id)
                    candidates = [primary] if primary is not None else []
                if candidates:
                    entries.append((len(candidates), net_id, pad_id, candidates))

        self._escape_records.clear()
        self._escape_spatial.clear()
        assignment = {}

        # Preserve a geometry planner's known-feasible primary pattern. The
        # remaining candidates are still available to each path search, but
        # there is no reason to spend a global min-conflicts pass rediscovering
        # a valid baseline.
        for _, net_id, pad_id, candidates in entries:
            assignment[pad_id] = candidates[0]
            self._insert_escape_record(self._escape_record(
                net_id, pad_id, candidates[0]
            ))
        primary_pairs, _, _ = self._detect_escape_conflicts()
        if not primary_pairs:
            self._escape_preferred_portals = assignment
            self._escape_assignment_conflicts = set()
            self._escape_reservations_strict = True
            self._escape_reserved_records = dict(self._escape_records)
            self._escape_reserved_spatial = defaultdict(set)
            for key, record in self._escape_reserved_records.items():
                for cell in record["cells"]:
                    self._escape_reserved_spatial[cell].add(key)
            self._escape_records.clear()
            self._escape_spatial.clear()
            logger.info(
                "[ESCAPE-ASSIGN] Preserved zero-conflict primary "
                "assignment for %d pads",
                len(assignment),
            )
            return

        self._escape_records.clear()
        self._escape_spatial.clear()
        assignment = {}

        # Constrained pads go first. Each later pad chooses the candidate
        # colliding with the fewest already assigned physical escapes.
        for _, net_id, pad_id, candidates in sorted(
            entries, key=lambda item: (item[0], item[2])
        ):
            ranked = []
            for portal in candidates:
                conflicts = self._escape_candidate_conflicts(
                    net_id, pad_id, portal
                )
                ranked.append((
                    conflicts,
                    self._portal_barrel_history.get(
                        (pad_id, portal.x_idx, portal.y_idx), 0.0
                    ),
                    float(getattr(portal, "score", 0.0)),
                    portal.y_idx,
                    portal,
                ))
            selected = min(ranked, key=lambda item: item[:4])[4]
            assignment[pad_id] = selected
            self._insert_escape_record(
                self._escape_record(net_id, pad_id, selected)
            )

        # Greedy insertion can paint a later pad into a corner. Standard
        # min-conflicts with sideways moves escapes the deterministic
        # two-variable swaps that occur in dense connector fields.
        import random

        pairs, _, _ = self._detect_escape_conflicts()
        adjacency = defaultdict(set)
        for first, second in pairs:
            adjacency[first].add(second)
            adjacency[second].add(first)
        conflict_keys = set(adjacency)
        best_pair_count = len(pairs)
        best_assignment = dict(assignment)
        rng = random.Random(42)
        key_cache = []
        max_steps = int(getattr(
            self.config, "escape_assignment_steps", 5000
        ))

        for step in range(max_steps):
            if not pairs:
                break
            if step % 128 == 0 or not key_cache:
                key_cache = list(conflict_keys)
            if not key_cache:
                break
            key = rng.choice(key_cache)
            if key not in conflict_keys:
                continue

            old_neighbors = tuple(adjacency.pop(key, ()))
            old = self._remove_escape_record(key)
            if old is None:
                conflict_keys.discard(key)
                continue
            for other in old_neighbors:
                adjacency[other].discard(key)
                pairs.discard(tuple(sorted((key, other))))
                if not adjacency[other]:
                    adjacency.pop(other, None)
                    conflict_keys.discard(other)
            conflict_keys.discard(key)

            candidates = self.portal_candidates.get(
                old["pad"], [old["portal"]]
            )
            ranked = []
            for portal in candidates:
                neighbors = self._escape_candidate_conflict_keys(
                    old["net"], old["pad"], portal
                )
                ranked.append((
                    len(neighbors),
                    self._portal_barrel_history.get(
                        (
                            old["pad"],
                            portal.x_idx,
                            portal.y_idx,
                        ),
                        0.0,
                    ),
                    float(getattr(portal, "score", 0.0)),
                    portal.y_idx,
                    portal,
                    neighbors,
                ))
            min_conflicts = min(item[0] for item in ranked)
            choices = [
                item for item in ranked if item[0] == min_conflicts
            ]
            if len(choices) > 1:
                alternatives = [
                    item for item in choices
                    if item[4] is not old["portal"]
                ]
                if alternatives:
                    choices = alternatives
            selected_item = rng.choice(choices)

            selected = selected_item[4]
            assignment[old["pad"]] = selected
            self._insert_escape_record(
                self._escape_record(
                    old["net"], old["pad"], selected
                )
            )
            for other in selected_item[5]:
                pair = tuple(sorted((key, other)))
                pairs.add(pair)
                adjacency[key].add(other)
                adjacency[other].add(key)
                conflict_keys.add(other)
            if adjacency.get(key):
                conflict_keys.add(key)

            if len(pairs) < best_pair_count:
                best_pair_count = len(pairs)
                best_assignment = dict(assignment)

        if len(pairs) > best_pair_count:
            assignment = best_assignment
            self._escape_records.clear()
            self._escape_spatial.clear()
            for _, net_id, pad_id, _ in entries:
                self._insert_escape_record(self._escape_record(
                    net_id, pad_id, assignment[pad_id]
                ))
            pairs, _, _ = self._detect_escape_conflicts()

        # Geometry-native planners may provide a known-feasible primary
        # pattern plus flexible length alternatives. Never let heuristic
        # candidate exploration discard that zero-conflict baseline.
        if pairs:
            primary_assignment = {
                pad_id: candidates[0]
                for _, _, pad_id, candidates in entries
            }
            self._escape_records.clear()
            self._escape_spatial.clear()
            for _, net_id, pad_id, _ in entries:
                self._insert_escape_record(self._escape_record(
                    net_id,
                    pad_id,
                    primary_assignment[pad_id],
                ))
            primary_pairs, _, _ = self._detect_escape_conflicts()
            if len(primary_pairs) < len(pairs):
                assignment = primary_assignment
                pairs = primary_pairs
            else:
                self._escape_records.clear()
                self._escape_spatial.clear()
                for _, net_id, pad_id, _ in entries:
                    self._insert_escape_record(self._escape_record(
                        net_id,
                        pad_id,
                        assignment[pad_id],
                    ))

        self._escape_preferred_portals = assignment
        self._escape_assignment_conflicts = set(pairs)
        self._escape_reservations_strict = not pairs
        logger.info(
            "[ESCAPE-ASSIGN] Assigned %d pads with %d physical conflicts",
            len(assignment),
            len(pairs),
        )
        self._escape_reserved_records = dict(self._escape_records)
        self._escape_reserved_spatial = defaultdict(set)
        for key, record in self._escape_reserved_records.items():
            for cell in record["cells"]:
                self._escape_reserved_spatial[cell].add(key)
        self._escape_records.clear()
        self._escape_spatial.clear()

    def _get_pad_portal_seeds(
        self, pad_id: str, current_net: str = None
    ) -> Tuple[List[Tuple[int, float]], Dict[int, Portal]]:
        """Expand every XY escape candidate across legal entry layers."""
        primary = self.portals.get(pad_id)
        candidates = self.portal_candidates.get(pad_id)
        if not candidates:
            candidates = [primary] if primary is not None else []
        fixed_layer = None
        if (
            current_net is not None
            and getattr(self, "_freeze_selected_portals", False)
            and current_net not in getattr(
                self, "_portal_cleanup_movable_nets", ()
            )
        ):
            pad_ids = self.net_pad_ids.get(current_net, ())
            selected = self.net_selected_portals.get(current_net, ())
            selected_layers = self.net_portal_layers.get(
                current_net, ()
            )
            if pad_id in pad_ids:
                position = pad_ids.index(pad_id)
                if (
                    position < len(selected)
                    and position < len(selected_layers)
                    and selected[position] is not None
                ):
                    candidates = [selected[position]]
                    fixed_layer = int(selected_layers[position])
        preferred = getattr(
            self, "_escape_preferred_portals", {}
        ).get(pad_id)

        best_by_node = {}
        portal_by_node = {}
        for portal in candidates:
            reserved_conflicts = 0
            if current_net is not None:
                committed_conflicts = self._escape_candidate_conflicts(
                    current_net, pad_id, portal
                )
                reserved_conflicts = self._escape_candidate_conflicts(
                    current_net,
                    pad_id,
                    portal,
                    records=self._escape_reserved_records,
                    spatial=self._escape_reserved_spatial,
                )
                # When the global assignment is exact, preserve it as a hard
                # physical reservation. Otherwise both reserved and already
                # committed escape geometry remain expensive negotiated
                # congestion. Making committed geometry a wall can eliminate
                # every seed and strand the net before Pathfinder can rip the
                # conflicting owner.
                if (
                    getattr(self, "_escape_reservations_strict", False)
                    and reserved_conflicts
                ):
                    continue
            candidate_penalty = float(getattr(portal, "score", 0.0))
            if current_net is not None:
                candidate_penalty += (
                    self._escape_candidate_congestion_penalty(
                        committed_conflicts,
                        reserved_conflicts,
                    )
                )
            candidate_penalty += (
                self._portal_barrel_history[
                    (pad_id, portal.x_idx, portal.y_idx)
                ]
                * float(getattr(
                    self.config,
                    "portal_barrel_history_penalty",
                    25.0,
                ))
                * float(getattr(self, "_pres_fac_now", 1.0))
            )
            if preferred is not None and portal is not preferred:
                candidate_penalty += float(getattr(
                    self.config, "escape_preference_penalty", 1.0
                ))
            for node, layer_cost in self._get_portal_seeds(portal):
                entry_layer = self.lattice.idx_to_coord(node)[2]
                if (
                    fixed_layer is not None
                    and entry_layer != fixed_layer
                ):
                    continue
                depth_history_penalty = (
                    self._portal_barrel_history[
                        (
                            pad_id,
                            portal.x_idx,
                            portal.y_idx,
                            entry_layer,
                        )
                    ]
                    * float(getattr(
                        self.config,
                        "portal_barrel_history_penalty",
                        25.0,
                    ))
                    * float(getattr(
                        self, "_pres_fac_now", 1.0
                    ))
                )
                total_cost = (
                    layer_cost
                    + candidate_penalty
                    + depth_history_penalty
                    + self._portal_chain_node_penalty(
                        portal,
                        entry_layer,
                        current_net,
                    )
                )
                if (
                    node not in best_by_node
                    or total_cost < best_by_node[node]
                ):
                    best_by_node[node] = total_cost
                    portal_by_node[node] = portal

        seeds = sorted(best_by_node.items())
        return seeds, portal_by_node

    def _escape_candidate_congestion_penalty(
        self,
        committed_conflicts: int,
        reserved_conflicts: int,
    ) -> float:
        """Price live escape copper more strongly during physical cleanup."""
        reservation_weight = float(getattr(
            self.config,
            "escape_reservation_penalty",
            1000.0,
        ))
        committed_weight = reservation_weight
        if getattr(self, "_freeze_selected_portals", False):
            committed_weight = max(
                committed_weight,
                float(getattr(
                    self.config,
                    "portal_cleanup_escape_penalty",
                    1_000_000.0,
                )),
            )
        return (
            float(committed_conflicts) * committed_weight
            + float(reserved_conflicts) * reservation_weight
        )

    def _portal_chain_node_penalty(
        self,
        portal: Portal,
        entry_layer: int,
        current_net: Optional[str],
    ) -> float:
        """Price the local escape barrel hidden behind a terminal seed.

        Multi-source routing contracts each pad's short F.Cu escape and blind
        barrel into a terminal edge. The graph search sees the entry node, but
        the intermediate barrel nodes are attached only after backtrace. Add
        their ownership, occupancy, and learned-history costs to the terminal
        edge so choosing an escape is equivalent to traversing that constrained
        local graph.
        """
        if current_net is None or not hasattr(self, "node_owner"):
            return 0.0

        pad_layer = int(portal.pad_layer)
        entry_layer = int(entry_layer)
        if pad_layer == entry_layer:
            return 0.0

        step = 1 if entry_layer > pad_layer else -1
        layers = np.arange(
            pad_layer,
            entry_layer,
            step,
            dtype=np.int32,
        )
        if layers.size == 0:
            return 0.0

        plane = self.lattice.x_steps * self.lattice.y_steps
        xy = portal.y_idx * self.lattice.x_steps + portal.x_idx
        nodes = layers.astype(np.int64) * plane + xy
        current_net_id = self._get_net_id(current_net)
        owners = self.node_owner[nodes]
        foreign = (owners != -1) & (owners != current_net_id)
        clearance_nodes = self._portal_clearance_halo_nodes(
            portal, entry_layer
        )
        clearance_owners = self.node_owner[clearance_nodes]
        clearance_portal_owners = self.portal_clearance_owner[
            clearance_nodes
        ]
        clearance_foreign = (
            ((clearance_owners != -1)
             & (clearance_owners != current_net_id))
            | ((clearance_portal_owners != -1)
               & (clearance_portal_owners != current_net_id))
        )

        pres_fac = float(getattr(self, "_pres_fac_now", 1.0))
        owner_weight = (
            float(getattr(self.config, "owner_penalty_base", 25.0))
            * pres_fac
        )
        path_weight = (
            float(getattr(self.config, "path_node_penalty_base", 25.0))
            * pres_fac
        )
        history_weight = float(getattr(
            self.config, "node_history_penalty", 5.0
        ))
        return float(
            foreign.sum() * owner_weight
            + (self.path_node_use[nodes] > 0).sum() * path_weight
            + self.node_conflict_history[nodes].sum() * history_weight
            + clearance_foreign.sum() * owner_weight
            + (
                self.path_node_use[clearance_nodes] > 0
            ).sum() * path_weight
            + self.node_conflict_history[
                clearance_nodes
            ].sum() * history_weight
        )

    def _attach_portal_vias(
        self,
        path: List[int],
        src_portal: Portal,
        dst_portal: Portal,
    ) -> List[int]:
        """Connect chosen internal portal seeds back to their F.Cu stubs."""
        if not path:
            return path

        _, _, src_layer = self.lattice.idx_to_coord(path[0])
        _, _, dst_layer = self.lattice.idx_to_coord(path[-1])
        src_step = 1 if src_layer > src_portal.pad_layer else -1
        dst_step = 1 if dst_portal.pad_layer > dst_layer else -1
        prefix = [
            self.lattice.node_idx(
                src_portal.x_idx, src_portal.y_idx, layer
            )
            for layer in range(
                src_portal.pad_layer, src_layer, src_step
            )
        ]
        suffix = [
            self.lattice.node_idx(
                dst_portal.x_idx, dst_portal.y_idx, layer
            )
            for layer in range(
                dst_layer + dst_step,
                dst_portal.pad_layer + dst_step,
                dst_step,
            )
        ]
        return prefix + path + suffix

    def _build_routing_seeds(self, portal_seeds_list):
        """
        Convert portal seeds from (node_idx, cost) tuples to plain node_idx arrays.

        Args:
            portal_seeds_list: List of (node_idx, cost) tuples from _get_portal_seeds()

        Returns:
            np.int32 array of node indices
        """
        import numpy as np
        if not portal_seeds_list:
            return np.array([], dtype=np.int32)
        # Extract just the node indices, ignore costs
        return np.array([seed[0] for seed in portal_seeds_list], dtype=np.int32)
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
        self._physical_cleanup_started = False
        self._previous_physical_conflict_count = None
        gpu_threshold = getattr(self.config, 'gpu_roi_min_nodes', 1000)
        logger.info(f"[GPU-THRESHOLD] GPU pathfinding enabled for ROIs with > {gpu_threshold} nodes")

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

                        # CRITICAL FIX: Use portal's computed entry_layer (routing layer, NOT F.Cu!)
                        # Portals land on internal routing layers (1-17), NOT on F.Cu (0)
                        # The escape planner already handles F.Cu → portal transitions
                        entry_layer = p1_portal.entry_layer  # Use portal's routing layer
                        exit_layer = p2_portal.entry_layer   # Use portal's routing layer

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

    def _accumulate_via_usage_for_path(
        self,
        node_path: List[int],
        net_id: str = None,
        *,
        col_use=None,
        seg_use=None,
    ):
        """
        Accumulate via column and segment usage for a committed path.
        Also registers via keepouts to prevent other nets from routing tracks through via locations.
        """
        if not hasattr(self, 'via_col_use') and not hasattr(self, 'via_seg_use'):
            return  # Pooling not enabled

        # Ensure via_keepouts_map exists
        if not hasattr(self, '_via_keepouts_map'):
            self._via_keepouts_map = {}

        idx_to_coord = self.lattice.idx_to_coord
        if col_use is None and hasattr(self, 'via_col_use'):
            col_use = self.via_col_use
        if seg_use is None and hasattr(self, 'via_seg_use'):
            seg_use = self.via_seg_use
        col_pool = col_use is not None
        seg_pool = seg_use is not None

        previous_via_xy = None
        for u, v in zip(node_path, node_path[1:]):
            xu, yu, zu = idx_to_coord(u)
            xv, yv, zv = idx_to_coord(v)

            # Check if it's a vertical transition (same x,y, different z)
            if xu == xv and yu == yv and zu != zv:
                # Adjacent-span graphs encode one physical barrel as a chain
                # of vertical hops. Count that contiguous chain once in the
                # column pool, while retaining per-segment occupancy below.
                via_xy = (xu, yu)
                if col_pool and via_xy != previous_via_xy:
                    col_use[xu, yu] += 1
                previous_via_xy = via_xy

                # Segment pooling: increment each segment crossed
                if seg_pool:
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    # Clamp to routing layers: 1..Nz-2
                    z_lo = max(1, min(z_lo, self._Nz - 2))
                    z_hi = max(1, min(z_hi, self._Nz - 2))
                    # Increment each segment z→z+1 in range [z_lo, z_hi)
                    for z in range(z_lo, z_hi):
                        seg_idx = z - 1  # Segment z→z+1 stored at index z-1
                        if 0 <= seg_idx < self._segZ:
                            seg_use[xu, yu, seg_idx] += 1

                # Register via keepouts for ALL layers the via touches (including endpoints!)
                # This prevents other nets from routing tracks through via locations
                if net_id:
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    for z in range(z_lo, z_hi + 1):  # Include both endpoints!
                        key = (z, xu, yu)
                        # First owner wins
                        if key not in self._via_keepouts_map:
                            self._via_keepouts_map[key] = net_id
            else:
                previous_via_xy = None

    def _rebuild_via_usage_from_committed(self):
        """Rebuild via column/segment usage from all currently committed net paths"""
        if not hasattr(self, 'via_col_use') and not hasattr(self, 'via_seg_use'):
            return

        # Accumulate on the host and upload once. Per-via writes into CuPy
        # arrays serialize the CPU and GPU thousands of times on large boards.
        col_use_cpu = (
            np.zeros((self._Nx, self._Ny), dtype=np.int16)
            if hasattr(self, 'via_col_use') else None
        )
        seg_use_cpu = (
            np.zeros((self._Nx, self._Ny, self._segZ), dtype=np.int16)
            if hasattr(self, 'via_seg_use') else None
        )

        # Clear routing via keepouts but PRESERVE portal keepouts
        # Portal keepouts were pre-registered to protect escape via columns
        if hasattr(self, '_via_keepouts_map'):
            # Save portal keepouts (any keepout not from a routed path)
            portal_keepouts = {k: v for k, v in self._via_keepouts_map.items()
                             if v not in self.net_paths}
            self._via_keepouts_map.clear()
            self._via_keepouts_map.update(portal_keepouts)
            if portal_keepouts:
                logger.debug(f"[VIA-REBUILD] Preserved {len(portal_keepouts)} portal keepouts during rebuild")

        # Rebuild from all committed paths (including keepout registration)
        for net_id, node_path in self.net_paths.items():
            if node_path and len(node_path) > 1:
                self._accumulate_via_usage_for_path(
                    node_path,
                    net_id=net_id,
                    col_use=col_use_cpu,
                    seg_use=seg_use_cpu,
                )

        # Escape barrels are attached to every committed routed path. Track
        # only geometry belonging to nets without a path, otherwise the same
        # physical barrel is counted twice.
        committed_nets = {
            net_id for net_id, path in self.net_paths.items() if path
        }
        self._track_escape_vias_in_via_usage(
            exclude_nets=committed_nets,
            col_use=col_use_cpu,
            seg_use=seg_use_cpu,
        )

        if col_use_cpu is not None:
            self.via_col_use[:] = self.accounting.xp.asarray(col_use_cpu)
        if seg_use_cpu is not None:
            self.via_seg_use[:] = self.accounting.xp.asarray(seg_use_cpu)

        # Log keepout statistics
        if hasattr(self, '_via_keepouts_map'):
            logger.info(f"[VIA-KEEPOUTS] Registered {len(self._via_keepouts_map)} via keepout cells")

        # REBUILD NODE OWNERSHIP (the correct solution!)
        self._rebuild_node_owner()
        self._rebuild_path_node_use()

    @staticmethod
    def _unique_path_nodes(path: List[int]) -> np.ndarray:
        if not path:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.asarray(path, dtype=np.int32))

    def _rebuild_path_node_use(self) -> None:
        """Rebuild soft node occupancy from all committed routes."""
        if not hasattr(self, "path_node_use"):
            return
        self.path_node_use.fill(0)
        for path in self.net_paths.values():
            nodes = self._unique_path_nodes(path)
            if nodes.size:
                self.path_node_use[nodes] += 1
        if self.path_node_use_gpu is not None:
            self.path_node_use_gpu[:] = cp.asarray(self.path_node_use)

    def _mark_path_node_use(self, path: List[int]) -> None:
        """Make a newly committed route visible to later routes."""
        nodes = self._unique_path_nodes(path)
        if not nodes.size:
            return
        self.path_node_use[nodes] += 1
        if self.path_node_use_gpu is not None:
            nodes_gpu = cp.asarray(nodes)
            self.path_node_use_gpu[nodes_gpu] = cp.asarray(
                self.path_node_use[nodes]
            )

    def _clear_path_node_use(self, path: List[int]) -> None:
        """Remove a ripped route from soft node occupancy."""
        nodes = self._unique_path_nodes(path)
        if not nodes.size:
            return
        self.path_node_use[nodes] = np.maximum(
            self.path_node_use[nodes] - 1, 0
        )
        if self.path_node_use_gpu is not None:
            nodes_gpu = cp.asarray(nodes)
            self.path_node_use_gpu[nodes_gpu] = cp.asarray(
                self.path_node_use[nodes]
            )

    def _rebuild_node_owner(self):
        """
        Rebuild node ownership map from portal reservations and committed vias.

        This is THE solution to via barrel conflicts:
        - Track which net owns each node (via barrels occupy nodes, not just edges)
        - Enforce via ROI bitmap masking (O(ROI_size) per net, not O(N×M)!)
        - Works for both full-graph and ROI routing

        Performance: O(portals + via_count × avg_span) = milliseconds, not minutes!
        """
        if not hasattr(self, 'node_owner'):
            return

        # Reset to all free
        self.node_owner.fill(-1)
        self._node_owner_members = {}
        self.portal_clearance_owner.fill(-1)
        self._portal_clearance_owner_members = {}

        owned_count = 0

        # 1. PORTAL RESERVATIONS: DISABLED - causing frontier empty issues
        # TODO: Debug why portal reservations block source seeds
        # if hasattr(self, 'portals') and self.portals:
        #     for pad_id, portal in self.portals.items():
        #         net_name = pad_id.rsplit('-', 1)[0] if '-' in pad_id else pad_id
        #         net_id = self._get_net_id(net_name)
        #         x_idx, y_idx = portal.x_idx, portal.y_idx
        #         for z in range(1, self.lattice.layers - 1):
        #             node_idx = self.lattice.node_idx(x_idx, y_idx, z)
        #             self.node_owner[node_idx] = net_id
        #             owned_count += 1

        # 2. COMMITTED VIA BARRELS: Mark nodes occupied by routed vias
        for net_name, path in self.net_paths.items():
            if not path or len(path) < 2:
                continue

            net_id = self._get_net_id(net_name)
            graph_path = self._path_without_dynamic_escape_chains(
                net_name, path
            )

            # Walk path and find layer transitions (vias)
            for i in range(len(graph_path) - 1):
                u, v = graph_path[i], graph_path[i+1]
                xu, yu, zu = self.lattice.idx_to_coord(u)
                xv, yv, zv = self.lattice.idx_to_coord(v)

                # Via: same (x,y), different z
                if xu == xv and yu == yv and zu != zv:
                    # Mark ALL nodes in the via barrel span
                    for node_idx in self._via_nodes_for_hop(u, v):
                        self._node_owner_members.setdefault(
                            node_idx, set()
                        ).add(net_id)

            for node_idx in self._selected_portal_clearance_nodes(
                net_name
            ):
                self._portal_clearance_owner_members.setdefault(
                    node_idx, set()
                ).add(net_id)

        for node_idx, members in self._node_owner_members.items():
            self.node_owner[node_idx] = (
                next(iter(members)) if len(members) == 1 else -2
            )
        owned_count = len(self._node_owner_members)
        for node_idx, members in (
            self._portal_clearance_owner_members.items()
        ):
            self.portal_clearance_owner[node_idx] = (
                next(iter(members)) if len(members) == 1 else -2
            )

        logger.info(
            "[NODE-OWNER] Marked %s graph-barrel and %s terminal-via "
            "clearance nodes as owned",
            f"{owned_count:,}",
            f"{len(self._portal_clearance_owner_members):,}",
        )
        if self.node_owner_gpu is not None:
            self.node_owner_gpu[:] = cp.asarray(self.node_owner)
        if self.portal_clearance_owner_gpu is not None:
            self.portal_clearance_owner_gpu[:] = cp.asarray(
                self.portal_clearance_owner
            )

    def _get_net_id(self, net_name: str) -> int:
        """Map net name to integer ID for node ownership"""
        if net_name not in self.net_id_map:
            self.net_id_map[net_name] = self.next_net_id
            self.next_net_id += 1
        return self.net_id_map[net_name]

    def _ensure_edge_src_map(self):
        """
        Build mapping from edge index to source node (once per routing).

        This is critical for barrel conflict detection - we need to know both
        endpoints of each edge. The CSR graph gives us destinations (indices),
        but we need to reconstruct sources from indptr.

        Performance: O(num_edges), done once at start of routing.
        """
        if hasattr(self, "_edge_src"):
            return

        import numpy as np

        # Get indptr from graph (handle both CPU and GPU)
        indptr = self.graph.indptr
        if hasattr(indptr, 'get'):
            indptr = indptr.get()

        # Total number of edges
        num_edges = len(self.graph.indices)

        # Build reverse mapping: edge_idx → source_node
        edge_src = np.empty(num_edges, dtype=np.int32)

        for u in range(len(indptr) - 1):
            edge_start = int(indptr[u])
            edge_end = int(indptr[u + 1])
            edge_src[edge_start:edge_end] = u

        self._edge_src = edge_src
        logger.info(f"[EDGE-SRC-MAP] Built mapping for {num_edges:,} edges")

    def _canonical_edge_resource_mask(self) -> np.ndarray:
        """Select one CSR arc for each undirected physical segment."""
        cached = getattr(
            self, "_canonical_edge_resource_mask_cache", None
        )
        if cached is not None:
            return cached
        self._ensure_edge_src_map()
        destinations = self.graph.indices
        if hasattr(destinations, "get"):
            destinations = destinations.get()
        mask = self._edge_src < np.asarray(destinations)
        self._canonical_edge_resource_mask_cache = mask
        return mask

    def _mark_via_barrel_ownership_for_path(self, net_name: str, path: List[int]) -> None:
        """
        Mark via barrel nodes as owned by this net IMMEDIATELY after commit.

        CRITICAL: This must be called AFTER each net commits, not just at iteration start!
        Without this, later nets in the same iteration don't see earlier nets' via barrels.
        """
        if not path or len(path) < 2:
            return

        net_id = self._get_net_id(net_name)
        graph_path = self._path_without_dynamic_escape_chains(
            net_name, path
        )
        owned_nodes = self._via_nodes_for_path(graph_path)
        for node_idx in owned_nodes:
            members = self._node_owner_members.setdefault(
                node_idx, set()
            )
            members.add(net_id)
            self.node_owner[node_idx] = (
                net_id if len(members) == 1 else -2
            )
        portal_nodes = self._selected_portal_clearance_nodes(net_name)
        for node_idx in portal_nodes:
            members = self._portal_clearance_owner_members.setdefault(
                node_idx, set()
            )
            members.add(net_id)
            self.portal_clearance_owner[node_idx] = (
                net_id if len(members) == 1 else -2
            )

        if owned_nodes and self.node_owner_gpu is not None:
            owned_nodes_gpu = cp.asarray(
                owned_nodes, dtype=cp.int32
            )
            self.node_owner_gpu[owned_nodes_gpu] = cp.asarray(
                self.node_owner[owned_nodes], dtype=cp.int32
            )
        if portal_nodes and self.portal_clearance_owner_gpu is not None:
            portal_nodes_gpu = cp.asarray(
                portal_nodes, dtype=cp.int32
            )
            self.portal_clearance_owner_gpu[
                portal_nodes_gpu
            ] = cp.asarray(
                self.portal_clearance_owner[portal_nodes],
                dtype=cp.int32,
            )

    def _clear_via_barrel_ownership_for_path(
        self, net_name: str, path: List[int]
    ) -> None:
        """Remove a ripped-up path without leaving ghost barrel owners."""
        if not path or len(path) < 2:
            return
        net_id = self._get_net_id(net_name)
        graph_path = self._path_without_dynamic_escape_chains(
            net_name, path
        )
        changed_nodes = self._via_nodes_for_path(graph_path)
        for node_idx in changed_nodes:
            members = self._node_owner_members.get(node_idx)
            if not members:
                self.node_owner[node_idx] = -1
                continue
            members.discard(net_id)
            if not members:
                self._node_owner_members.pop(node_idx, None)
                self.node_owner[node_idx] = -1
            elif len(members) == 1:
                self.node_owner[node_idx] = next(iter(members))
            else:
                self.node_owner[node_idx] = -2
        portal_nodes = self._selected_portal_clearance_nodes(net_name)
        for node_idx in portal_nodes:
            members = self._portal_clearance_owner_members.get(node_idx)
            if not members:
                self.portal_clearance_owner[node_idx] = -1
                continue
            members.discard(net_id)
            if not members:
                self._portal_clearance_owner_members.pop(node_idx, None)
                self.portal_clearance_owner[node_idx] = -1
            elif len(members) == 1:
                self.portal_clearance_owner[node_idx] = next(iter(members))
            else:
                self.portal_clearance_owner[node_idx] = -2

        if changed_nodes and self.node_owner_gpu is not None:
            nodes_gpu = cp.asarray(changed_nodes, dtype=cp.int32)
            self.node_owner_gpu[nodes_gpu] = cp.asarray(
                self.node_owner[changed_nodes], dtype=cp.int32
            )
        if portal_nodes and self.portal_clearance_owner_gpu is not None:
            nodes_gpu = cp.asarray(portal_nodes, dtype=cp.int32)
            self.portal_clearance_owner_gpu[nodes_gpu] = cp.asarray(
                self.portal_clearance_owner[portal_nodes],
                dtype=cp.int32,
            )

    def _selected_portal_clearance_nodes(
        self, net_name: str
    ) -> List[int]:
        """Return graph nodes whose copper would hit a net's terminal vias."""
        selected = self.net_selected_portals.get(net_name, ())
        layers = self.net_portal_layers.get(net_name, ())
        nodes = set()
        for portal, entry_layer in zip(selected, layers):
            if portal is not None:
                nodes.update(self._portal_clearance_nodes(
                    portal, int(entry_layer)
                ))
        return sorted(nodes)

    def _portal_clearance_nodes(
        self, portal: Portal, entry_layer: int
    ) -> np.ndarray:
        """Map one off-grid terminal-via envelope onto affected graph nodes."""
        cache_key = (id(portal), int(entry_layer))
        cached = self._portal_clearance_nodes_cache.get(cache_key)
        if cached is not None:
            return cached

        xy_by_axis = self._portal_clearance_xy_nodes(portal)
        plane = self.lattice.x_steps * self.lattice.y_steps
        z_lo, z_hi = sorted((
            int(portal.pad_layer), int(entry_layer)
        ))
        chunks = []
        for layer in range(z_lo, z_hi + 1):
            xy_nodes = xy_by_axis["via"]
            if not (
                self.lattice.layers > 2
                and layer in (0, self.lattice.layers - 1)
            ):
                for axis in self.lattice.get_allowed_axes(layer):
                    xy_nodes = np.union1d(
                        xy_nodes, xy_by_axis[axis]
                    )
            if xy_nodes.size:
                chunks.append(xy_nodes + layer * plane)

        result = (
            np.unique(np.concatenate(chunks)).astype(
                np.int32, copy=False
            )
            if chunks else np.empty(0, dtype=np.int32)
        )
        self._portal_clearance_nodes_cache[cache_key] = result
        return result

    def _portal_clearance_halo_nodes(
        self, portal: Portal, entry_layer: int
    ) -> np.ndarray:
        """Clearance footprint excluding the contracted centerline chain."""
        cache_key = (id(portal), int(entry_layer))
        cached = self._portal_clearance_halo_cache.get(cache_key)
        if cached is not None:
            return cached
        nodes = self._portal_clearance_nodes(portal, entry_layer)
        pad_layer = int(portal.pad_layer)
        step = 1 if entry_layer > pad_layer else -1
        plane = self.lattice.x_steps * self.lattice.y_steps
        xy = portal.y_idx * self.lattice.x_steps + portal.x_idx
        chain = (
            np.arange(
                pad_layer, entry_layer, step, dtype=np.int64
            ) * plane + xy
        )
        result = np.setdiff1d(
            nodes, chain, assume_unique=False
        ).astype(np.int32, copy=False)
        self._portal_clearance_halo_cache[cache_key] = result
        return result

    def _portal_clearance_xy_nodes(self, portal: Portal):
        """Compute reusable XY clearance templates for one portal."""
        cache_key = id(portal)
        cached = self._portal_clearance_xy_cache.get(cache_key)
        if cached is not None:
            return cached

        center = self.escape_planner._portal_world(portal)
        pitch = float(self.lattice.geom.pitch)
        via_track_limit = max(
            0.5 * float(self.config.via_diameter)
            + 0.5 * float(self.config.track_width)
            + float(self.config.clearance),
            0.5 * float(self.config.via_drill)
            + 0.5 * float(self.config.track_width)
            + float(getattr(self.config, "hole_clearance", 0.0)),
        )
        via_via_limit = max(
            float(self.config.via_diameter)
            + float(self.config.clearance),
            float(self.config.via_drill)
            + float(getattr(
                self.config, "min_hole_to_hole", 0.0
            )),
        )
        search_steps = int(np.ceil(
            max(via_track_limit, via_via_limit) / pitch
        )) + 1
        x_lo = max(0, portal.x_idx - search_steps)
        x_hi = min(
            self.lattice.x_steps - 1,
            portal.x_idx + search_steps,
        )
        y_lo = max(0, portal.y_idx - search_steps)
        y_hi = min(
            self.lattice.y_steps - 1,
            portal.y_idx + search_steps,
        )
        via_xy = set()
        for x_idx in range(x_lo, x_hi + 1):
            for y_idx in range(y_lo, y_hi + 1):
                world = self.lattice.geom.lattice_to_world(
                    x_idx, y_idx
                )
                if (
                    float(np.hypot(
                        world[0] - center[0],
                        world[1] - center[1],
                    ))
                    < via_via_limit - 1e-9
                ):
                    via_xy.add(
                        y_idx * self.lattice.x_steps + x_idx
                    )

        axis_xy = {}
        for axis in ("h", "v"):
            track_xy = set()
            if axis == "h":
                segments = (
                    (x_idx, y_idx, x_idx + 1, y_idx)
                    for y_idx in range(y_lo, y_hi + 1)
                    for x_idx in range(x_lo, x_hi)
                )
            else:
                segments = (
                    (x_idx, y_idx, x_idx, y_idx + 1)
                    for x_idx in range(x_lo, x_hi + 1)
                    for y_idx in range(y_lo, y_hi)
                )
            for x0, y0, x1, y1 in segments:
                start = self.lattice.geom.lattice_to_world(x0, y0)
                end = self.lattice.geom.lattice_to_world(x1, y1)
                if (
                    self._point_segment_distance(center, start, end)
                    < via_track_limit - 1e-9
                ):
                    track_xy.update((
                        y0 * self.lattice.x_steps + x0,
                        y1 * self.lattice.x_steps + x1,
                    ))
            axis_xy[axis] = np.asarray(
                sorted(track_xy), dtype=np.int32
            )

        result = {
            "via": np.asarray(sorted(via_xy), dtype=np.int32),
            **axis_xy,
        }
        self._portal_clearance_xy_cache[cache_key] = result
        return result

    def _via_nodes_for_hop(self, u: int, v: int) -> List[int]:
        xu, yu, zu = self.lattice.idx_to_coord(u)
        xv, yv, zv = self.lattice.idx_to_coord(v)
        if xu != xv or yu != yv or zu == zv:
            return []
        z_lo, z_hi = sorted((zu, zv))
        return [
            self.lattice.node_idx(xu, yu, z)
            for z in range(z_lo, z_hi + 1)
        ]

    def _via_nodes_for_path(self, path: List[int]) -> List[int]:
        nodes = set()
        for u, v in zip(path, path[1:]):
            nodes.update(self._via_nodes_for_hop(u, v))
        return sorted(nodes)

    def _build_owner_bitmap_for_fullgraph(self, current_net: str, force_allow_nodes=None) -> np.ndarray:
        """
        Build owner-aware bitmap for full-graph routing.

        Returns uint32 bitmap where bit=1 if node is free OR owned by current net.
        This allows the GPU wavefront kernel to skip nodes owned by other nets.

        CRITICAL: Force-allows source/dest seeds even if ownership bookkeeping lags!

        Performance: O(N/32) vectorized bitmap operations = milliseconds per net
        Memory: ~14k uint32 words (~56KB) per bitmap

        Args:
            current_net: Net currently being routed
            force_allow_nodes: Source/dest nodes to force-allow (even if owned by others)

        Returns:
            uint32 bitmap array (words = ceil(num_nodes/32))
        """
        net_id = self._get_net_id(current_net)
        owners = self.node_owner  # np.int32[num_nodes]

        # Vectorized: allowed = (free OR owned by current net)
        allowed = (owners == -1) | (owners == net_id)

        # CRITICAL: Force-allow seeds (prevents frontier empty!)
        if force_allow_nodes is not None and len(force_allow_nodes) > 0:
            allowed[force_allow_nodes] = True

        n = int(allowed.size)
        words = (n + 31) // 32
        bitmap = np.zeros(words, dtype=np.uint32)

        # Pack bits into words (vectorized)
        idx = np.nonzero(allowed)[0].astype(np.int64)
        if len(idx) > 0:
            word_indices = (idx >> 5).astype(np.int32)  # idx // 32
            bit_positions = (idx & 31).astype(np.int32)  # idx % 32
            bit_values = (1 << bit_positions).astype(np.uint32)

            # OR bits into words (use add.at since bits don't overlap per index)
            np.add.at(bitmap, word_indices, bit_values)

        return bitmap

    def _build_owner_penalty(self, roi_nodes: Optional[np.ndarray],
                             current_net: str,
                             force_allow_nodes=None) -> Optional[np.ndarray]:
        """
        Build a ROI-local node penalty pricing nodes owned by OTHER nets.

        Ownership-as-cost: entering a foreign-owned node (a via barrel or
        escape column) costs owner_penalty_base * pres_fac on top of the edge
        cost. Early iterations can cross barrels cheaply (greedy connect);
        as pres_fac escalates, foreign barrels become prohibitively expensive
        and nets negotiate around them. This replaced a hard ROI filter that
        removed owned nodes entirely - which could strip a net's OWN
        endpoints and made barrel conflicts unresolvable by negotiation.

        Returns:
            float32 array (len == len(roi_nodes)) or None if nothing owned.
        """
        if not hasattr(self, 'node_owner'):
            return None

        current_net_id = self._get_net_id(current_net)
        owners = self.node_owner if roi_nodes is None else self.node_owner[roi_nodes]
        foreign = (owners != -1) & (owners != current_net_id)
        portal_owners = (
            self.portal_clearance_owner
            if roi_nodes is None
            else self.portal_clearance_owner[roi_nodes]
        )
        foreign_portal = (
            (portal_owners != -1)
            & (portal_owners != current_net_id)
        )
        path_use = (
            self.path_node_use
            if roi_nodes is None else self.path_node_use[roi_nodes]
        )
        occupied = path_use > 0
        node_history = (
            self.node_conflict_history
            if roi_nodes is None
            else self.node_conflict_history[roi_nodes]
        )
        if roi_nodes is None and force_allow_nodes is not None:
            foreign[force_allow_nodes] = False
            foreign_portal[force_allow_nodes] = False
            occupied[force_allow_nodes] = False
        if (
            not foreign.any()
            and not foreign_portal.any()
            and not occupied.any()
            and not node_history.any()
        ):
            return None

        weight = (float(getattr(self.config, 'owner_penalty_base', 25.0))
                  * float(getattr(self, '_pres_fac_now', 1.0)))
        if getattr(self, "_freeze_selected_portals", False):
            weight = max(
                weight,
                float(getattr(
                    self.config,
                    "portal_cleanup_node_penalty",
                    1_000_000.0,
                )),
            )
        path_weight = (
            float(getattr(self.config, 'path_node_penalty_base', 25.0))
            * float(getattr(self, '_pres_fac_now', 1.0))
        )
        history_weight = float(getattr(
            self.config, "node_history_penalty", 5.0
        ))
        return (
            (foreign | foreign_portal).astype(np.float32) * weight
            + occupied.astype(np.float32) * path_weight
            + node_history * history_weight
        )

    def _build_owner_penalty_gpu(self, current_net: str,
                                 force_allow_nodes=None):
        """Build the full-graph ownership cost without a host transfer."""
        if self.node_owner_gpu is None:
            return self._build_owner_penalty(
                None, current_net, force_allow_nodes=force_allow_nodes
            )

        current_net_id = self._get_net_id(current_net)
        weight = np.float32(
            float(getattr(self.config, 'owner_penalty_base', 25.0))
            * float(getattr(self, '_pres_fac_now', 1.0))
        )
        if getattr(self, "_freeze_selected_portals", False):
            weight = np.float32(max(
                float(weight),
                float(getattr(
                    self.config,
                    "portal_cleanup_node_penalty",
                    1_000_000.0,
                )),
            ))
        owners = self.node_owner_gpu
        penalty = cp.where(
            (owners != -1) & (owners != current_net_id),
            weight,
            cp.float32(0.0),
        ).astype(cp.float32, copy=False)
        if self.portal_clearance_owner_gpu is not None:
            portal_owners = self.portal_clearance_owner_gpu
            penalty = cp.maximum(
                penalty,
                cp.where(
                    (portal_owners != -1)
                    & (portal_owners != current_net_id),
                    weight,
                    cp.float32(0.0),
                ),
            )
        path_weight = np.float32(
            float(getattr(self.config, 'path_node_penalty_base', 25.0))
            * float(getattr(self, '_pres_fac_now', 1.0))
        )
        if self.path_node_use_gpu is not None and path_weight > 0:
            penalty += cp.where(
                self.path_node_use_gpu > 0,
                path_weight,
                cp.float32(0.0),
            )
        history_weight = np.float32(getattr(
            self.config, "node_history_penalty", 5.0
        ))
        if (
            self.node_conflict_history_gpu is not None
            and history_weight > 0
        ):
            penalty += (
                self.node_conflict_history_gpu * history_weight
            )
        if force_allow_nodes is not None and len(force_allow_nodes) > 0:
            penalty[cp.asarray(force_allow_nodes, dtype=cp.int32)] = 0.0
        return penalty

    def _track_escape_vias_in_via_usage(
        self,
        exclude_nets=None,
        *,
        col_use=None,
        seg_use=None,
    ):
        """
        Register escape vias in via spatial tracking arrays.

        This ensures that via columns and segments used by pad escape vias
        are properly tracked, preventing routing conflicts.
        """
        if not hasattr(self, '_escape_vias') or not self._escape_vias:
            return

        if not hasattr(self, 'via_col_use') and not hasattr(self, 'via_seg_use'):
            # Via spatial tracking not enabled
            return

        tracked_count = 0
        exclude_nets = set(exclude_nets or ())
        if col_use is None and hasattr(self, 'via_col_use'):
            col_use = self.via_col_use
        if seg_use is None and hasattr(self, 'via_seg_use'):
            seg_use = self.via_seg_use

        for via_dict in self._escape_vias:
            if via_dict.get('net') in exclude_nets:
                continue
            # Extract via information
            x_mm = via_dict.get('x')
            y_mm = via_dict.get('y')
            from_layer_name = via_dict.get('from_layer')
            to_layer_name = via_dict.get('to_layer')

            if x_mm is None or y_mm is None or from_layer_name is None or to_layer_name is None:
                continue

            # Convert world coordinates to lattice indices
            xu, yu = self.lattice.world_to_lattice(x_mm, y_mm)
            if xu < 0 or xu >= self.lattice.x_steps or yu < 0 or yu >= self.lattice.y_steps:
                continue

            # Convert layer names to indices
            z_lo = self._layer_name_to_index(from_layer_name)
            z_hi = self._layer_name_to_index(to_layer_name)

            if z_lo is None or z_hi is None:
                continue

            # Ensure z_lo < z_hi
            if z_lo > z_hi:
                z_lo, z_hi = z_hi, z_lo

            # Track in column usage
            if col_use is not None:
                col_use[xu, yu] += 1

            # Track in segment usage
            if seg_use is not None:
                for z in range(z_lo, z_hi):
                    seg_idx = z - 1  # Segments indexed from 0
                    if 0 <= seg_idx < self._segZ:
                        seg_use[xu, yu, seg_idx] += 1

            # Register via keepouts for ALL layers (including endpoints) to block tracks
            if not hasattr(self, '_via_keepouts_map'):
                self._via_keepouts_map = {}

            net_id = via_dict.get('net', 'escape_via')
            for z in range(z_lo, z_hi + 1):  # Include both endpoints!
                key = (z, xu, yu)
                self._via_keepouts_map.setdefault(key, net_id)

            tracked_count += 1

        if tracked_count > 0:
            logger.info(f"[ESCAPE-VIA] Tracked {tracked_count} escape vias in via spatial arrays")

    def _layer_name_to_index(self, layer_name) -> Optional[int]:
        """Convert layer name to layer index, or None if not found"""
        if isinstance(layer_name, (int, np.integer)):
            layer = int(layer_name)
            return (
                layer
                if 0 <= layer < int(self.config.layer_count)
                else None
            )
        if not hasattr(self.config, 'layer_names'):
            return None

        try:
            return self.config.layer_names.index(layer_name)
        except (ValueError, AttributeError):
            # Try numeric layer format like "L5"
            if layer_name.startswith('L') and layer_name[1:].isdigit():
                return int(layer_name[1:])
            return None

    def _apply_via_keepouts_to_graph(self):
        """
        Apply via keepouts to the graph by blocking planar routing at via locations.

        This prevents tracks from routing through via locations when using full-graph routing.
        The via keepouts are stored in _via_keepouts_map as (z, x, y) -> net_id.

        For each keepout location, we block PLANAR edges (same-layer horizontal/vertical edges)
        but allow VIA edges (inter-layer edges) so that other vias can still use the same column.

        Note: This is a global blocking approach. For per-net owner-aware keepouts, see ROI extraction.
        This method is called after via usage tracking is rebuilt, ensuring all current vias are blocked.
        """
        if not hasattr(self, '_via_keepouts_map') or not self._via_keepouts_map:
            logger.debug("[VIA-KEEPOUT] No via keepouts to apply")
            return

        if not hasattr(self, 'graph') or self.graph is None:
            logger.warning("[VIA-KEEPOUT] Graph not initialized, cannot apply keepouts")
            return

        # Get the base cost array (this is where we'll block edges)
        if not hasattr(self.graph, 'base_costs') or self.graph.base_costs is None:
            logger.warning("[VIA-KEEPOUT] Base cost array not available, cannot apply keepouts")
            return

        base_cost = self.graph.base_costs
        is_gpu = hasattr(base_cost, 'get')  # Check if it's a GPU array

        # Convert to CPU for modification if needed
        if is_gpu:
            base_cost_cpu = base_cost.get()
        else:
            base_cost_cpu = base_cost

        # Get graph structure
        indptr = self.graph.indptr
        indices = self.graph.indices

        # Convert to CPU if on GPU
        if hasattr(indptr, 'get'):
            indptr = indptr.get()
        if hasattr(indices, 'get'):
            indices = indices.get()

        blocked_planar_edges = 0
        keepout_block_cost = 1e9  # Very high cost to effectively block the edge

        # For each via keepout location, block PLANAR edges only (not via edges)
        for (z, x, y), owner_net in self._via_keepouts_map.items():
            # Convert lattice coordinates to node index
            node_idx = self.lattice.node_idx(x, y, z)

            # Get source coordinates
            src_x, src_y, src_z = self.lattice.idx_to_coord(node_idx)

            # Block planar outgoing edges from this node
            start = int(indptr[node_idx])
            end = int(indptr[node_idx + 1])

            for edge_idx in range(start, end):
                dst_node = int(indices[edge_idx])
                dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst_node)

                # Only block PLANAR edges (same layer), allow VIA edges
                if src_z == dst_z:
                    # This is a planar edge (horizontal or vertical track)
                    base_cost_cpu[edge_idx] = keepout_block_cost
                    blocked_planar_edges += 1
                # Via edges (src_z != dst_z) are NOT blocked to allow other vias in same column

        # Update the base cost array (copy back to GPU if needed)
        if is_gpu:
            try:
                import cupy as cp
                self.graph.base_costs = cp.asarray(base_cost_cpu)
            except ImportError:
                self.graph.base_costs = base_cost_cpu
        else:
            self.graph.base_costs = base_cost_cpu

        logger.info(f"[VIA-KEEPOUT] Applied {len(self._via_keepouts_map)} via keepouts, blocked {blocked_planar_edges} planar edges in full graph")

    def _apply_owner_aware_via_keepouts(self, current_net_id: str, costs) -> int:
        """
        Apply via keepouts for full-graph routing (owner-aware).

        Temporarily blocks planar edges at via locations owned by OTHER nets.
        This prevents the current net from routing tracks through other nets' via barrels.

        Args:
            current_net_id: Net currently being routed
            costs: Cost array (CuPy or NumPy)

        Returns:
            Number of edges blocked
        """
        if not hasattr(self, '_via_keepouts_map') or not self._via_keepouts_map:
            return 0

        # Store original costs for restoration
        if not hasattr(self, '_via_keepout_backup'):
            self._via_keepout_backup = {}

        is_gpu = hasattr(costs, 'device')
        xp = cp if is_gpu else np

        # Get graph structure (already on appropriate device)
        indptr = self.graph.indptr
        indices = self.graph.indices

        blocked_count = 0
        keepout_block_cost = 1e9

        # Convert indptr/indices to CPU for indexing (they're small, cached once)
        if is_gpu:
            if not hasattr(self, '_indptr_cpu_cache'):
                self._indptr_cpu_cache = indptr.get()
                self._indices_cpu_cache = indices.get()
            indptr_cpu = self._indptr_cpu_cache
            indices_cpu = self._indices_cpu_cache
        else:
            indptr_cpu = indptr
            indices_cpu = indices

        # Block via locations owned by OTHER nets
        # CRITICAL: Via occupies NODE, so block ALL edges to/from that node!
        for (z, x, y), owner_net in self._via_keepouts_map.items():
            # Skip vias owned by current net (owner-aware!)
            if owner_net == current_net_id:
                continue

            # Get node index - this node is OCCUPIED by another net's via barrel
            node_idx = self.lattice.node_idx(x, y, z)

            # Block ALL OUTGOING edges from this via node (no routing through via!)
            start, end = int(indptr_cpu[node_idx]), int(indptr_cpu[node_idx + 1])

            for edge_idx in range(start, end):
                dst_node = int(indices_cpu[edge_idx])
                dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst_node)

                # Block ALL edges except vias in the same column (allow stacked vias)
                # Only allow via edges if both nodes in same (x,y) column
                if not (dst_x == x and dst_y == y and dst_z != z):
                    # This is either a planar edge OR a via to different (x,y) - block it!
                    if edge_idx not in self._via_keepout_backup:
                        if is_gpu:
                            self._via_keepout_backup[edge_idx] = float(costs[edge_idx])
                        else:
                            self._via_keepout_backup[edge_idx] = float(costs[edge_idx])

                    costs[edge_idx] = keepout_block_cost
                    blocked_count += 1

        # NOTE: Owner-aware blocking removed - O(N×M) doesn't scale to 16k+ keepouts
        # Causes 6+ second overhead per net (512 nets = 50+ minutes!)
        #if blocked_count > 0:
        #    logger.info(f"[VIA-KEEPOUT-OWNER] Blocked {blocked_count} edges for net {current_net_id}")

        return blocked_count

    def _restore_via_keepout_costs(self, costs):
        """Restore original costs after owner-aware via keepout blocking"""
        if not hasattr(self, '_via_keepout_backup') or not self._via_keepout_backup:
            return

        is_gpu = hasattr(costs, 'device')

        for edge_idx, original_cost in self._via_keepout_backup.items():
            costs[edge_idx] = original_cost

        # Clear backup for next net
        self._via_keepout_backup.clear()

    def _apply_via_pooling_penalties(self, pres_fac: float):
        """
        Apply via column and segment pooling penalties to vertical edge costs.

        Uses GPU-accelerated CUDA kernel when available (800ms → <2ms speedup!)
        Falls back to CPU vectorized implementation if GPU unavailable.
        """
        import time
        t0 = time.perf_counter()

        if not hasattr(self, 'via_col_pres') and not hasattr(self, 'via_seg_pres'):
            return

        # Check if metadata is available
        if not hasattr(self, '_via_edge_metadata') or self._via_edge_metadata is None:
            logger.warning("[VIA-POOL] Metadata not built, falling back to sequential")
            self._apply_via_pooling_penalties_sequential(pres_fac)
            return

        # Try GPU kernel first if available
        if hasattr(self, 'via_kernel_manager') and self.via_kernel_manager.use_gpu:
            try:
                # Check if there are any penalties to apply (GPU can check this fast)
                xp = cp if hasattr(self.via_col_pres, 'device') else np
                if hasattr(self, 'via_col_pres'):
                    col_max = float(xp.max(self.via_col_pres))
                    if col_max == 0 and hasattr(self, 'via_seg_pres'):
                        seg_max = float(xp.max(self.via_seg_pres))
                        if seg_max == 0:
                            return  # No penalties needed

                col_weight = float(getattr(self.config, "via_column_weight", 1.0))
                seg_weight = float(getattr(self.config, "via_segment_weight", 1.0))

                penalty_count = self.via_kernel_manager.apply_via_penalties(
                    via_metadata=self._via_edge_metadata,
                    via_col_pres_gpu=self.via_col_pres,
                    via_seg_pres_gpu=self.via_seg_pres if hasattr(self, 'via_seg_pres') else None,
                    col_weight=col_weight * pres_fac,
                    seg_weight=seg_weight * pres_fac,
                    total_cost_gpu=self.accounting.total_cost,
                    Ny=self._Ny,
                    segZ=self._segZ if hasattr(self, '_segZ') else 0
                )
                return  # GPU kernel succeeded
            except Exception as e:
                logger.warning(f"[VIA-POOL] GPU kernel failed: {e}, falling back to CPU")

        # CPU fallback - original vectorized implementation
        col_weight = float(getattr(self.config, "via_column_weight", 1.0))
        seg_weight = float(getattr(self.config, "via_segment_weight", 1.0))

        # Get cost array
        total_cost = self.accounting.total_cost
        if self.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get precomputed metadata
        via_edge_indices = self._via_edge_metadata['indices']
        via_xy_coords = self._via_edge_metadata['xy_coords']
        z_lo = self._via_edge_metadata['z_lo']
        z_hi = self._via_edge_metadata['z_hi']

        num_via_edges = len(via_edge_indices)
        if num_via_edges == 0:
            return

        # Initialize penalties array
        penalties = np.zeros(num_via_edges, dtype=np.float32)

        # Vectorized column penalty computation
        if hasattr(self, 'via_col_pres'):
            col_penalties = self.via_col_pres[via_xy_coords[:, 0], via_xy_coords[:, 1]]
            penalties += col_weight * col_penalties

        # Vectorized segment penalty computation (using prefix sums)
        if hasattr(self, 'via_seg_prefix'):
            # Compute prefix indices for range queries
            # Segment index mapping: z-1→z is stored at index z-2
            hi_idx = z_hi - 2  # Index for upper bound
            lo_idx = z_lo - 2  # Index for lower bound

            # Create masks for valid indices
            valid_mask = z_hi > z_lo  # Only process edges spanning multiple layers
            hi_valid = (hi_idx >= 0) & (hi_idx < self._segZ)
            lo_valid = (lo_idx >= 0) & (lo_idx < self._segZ)

            # Fetch prefix values with bounds checking
            pref_hi = np.zeros(num_via_edges, dtype=np.float32)
            pref_lo = np.zeros(num_via_edges, dtype=np.float32)

            # Use advanced indexing for valid entries
            if np.any(hi_valid):
                valid_hi_edges = hi_valid
                pref_hi[valid_hi_edges] = self.via_seg_prefix[
                    via_xy_coords[valid_hi_edges, 0],
                    via_xy_coords[valid_hi_edges, 1],
                    hi_idx[valid_hi_edges]
                ]

            if np.any(lo_valid):
                valid_lo_edges = lo_valid
                pref_lo[valid_lo_edges] = self.via_seg_prefix[
                    via_xy_coords[valid_lo_edges, 0],
                    via_xy_coords[valid_lo_edges, 1],
                    lo_idx[valid_lo_edges]
                ]

            # Compute segment penalties: prefix[hi] - prefix[lo]
            seg_penalties = (pref_hi - pref_lo) * valid_mask
            penalties += seg_weight * seg_penalties

        # STEP 2.7: Apply "leave-hot-layer" via discount using layer bias
        if hasattr(self, 'layer_bias'):
            k = float(getattr(self.config, 'via_hot_layer_discount', 0.20))

            # Get source and destination layer biases for each via edge
            src_bias = self.layer_bias[z_lo]
            dst_bias = self.layer_bias[z_hi]

            # Cheaper to leave hot layers, more expensive to land on hot layers
            via_discount = (1.0 - k * np.maximum(src_bias, 0.0)) * (1.0 + 0.5 * k * np.maximum(dst_bias, 0.0))
            penalties *= via_discount  # Apply discount/markup to penalties

            # Log discount statistics if significant
            avg_discount = float(np.mean(via_discount))
            if abs(avg_discount - 1.0) > 0.05:
                logger.debug(f"[VIA-DISCOUNT] Average via discount factor: {avg_discount:.3f}")

        # Apply penalties to cost array (vectorized)
        penalty_mask = penalties > 0
        total_cost_cpu[via_edge_indices[penalty_mask]] += pres_fac * penalties[penalty_mask]
        penalties_applied = np.sum(penalty_mask)

        # Update GPU if needed
        if self.accounting.use_gpu:
            self.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

        elapsed = time.perf_counter() - t0
        if penalties_applied > 0:
            logger.info(f"[VIA-POOL-PERF] Vectorized penalty application: {num_via_edges} edges, {penalties_applied} penalties in {elapsed:.3f}s")
        else:
            logger.debug(f"[VIA-POOL-PERF] No penalties applied ({num_via_edges} edges checked in {elapsed:.3f}s)")

    def _apply_via_pooling_penalties_sequential(self, pres_fac: float):
        """Sequential fallback for via pooling penalties (for debugging/comparison)"""
        col_weight = float(getattr(self.config, "via_column_weight", 1.0))
        seg_weight = float(getattr(self.config, "via_segment_weight", 1.0))

        # Get cost array
        total_cost = self.accounting.total_cost
        if self.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get graph data
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        idx_to_coord = self.lattice.idx_to_coord
        penalties_applied = 0

        # Find via edge indices (where _via_edges is True)
        via_edge_indices = np.where(self._via_edges)[0]

        for ei in via_edge_indices:
            u = int(np.searchsorted(indptr, ei, side='right') - 1)
            if 0 <= u < len(indptr) - 1 and indptr[u] <= ei < indptr[u + 1]:
                v = int(indices[ei])
                xu, yu, zu = idx_to_coord(u)
                xv, yv, zv = idx_to_coord(v)

                penalty = 0.0

                # Column penalty
                if hasattr(self, 'via_col_pres'):
                    penalty += col_weight * self.via_col_pres[xu, yu]

                # Segment penalty (use prefix for fast range sum)
                if hasattr(self, 'via_seg_prefix'):
                    z_lo, z_hi = (zu, zv) if zu < zv else (zv, zu)
                    z_lo = max(1, min(z_lo, self._Nz - 2))
                    z_hi = max(1, min(z_hi, self._Nz - 2))
                    if z_hi >= z_lo:  # Allow equal (single-segment vias)
                        hi_idx = z_hi - 2
                        lo_idx = z_lo - 2
                        pref_hi = self.via_seg_prefix[xu, yu, hi_idx] if 0 <= hi_idx < self._segZ else 0.0
                        pref_lo = self.via_seg_prefix[xu, yu, lo_idx] if 0 <= lo_idx < self._segZ else 0.0
                        seg_sum = pref_hi - pref_lo
                        penalty += seg_weight * seg_sum

                if penalty > 0:
                    total_cost_cpu[ei] += pres_fac * penalty
                    penalties_applied += 1

        # Update GPU if needed
        if self.accounting.use_gpu:
            self.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

        if penalties_applied > 0:
            logger.debug(f"[VIA-POOL] Sequential: Applied pooling penalties to {penalties_applied} via edges")

    def _spatial_via_overuse_total(self) -> int:
        """Return over-capacity via-column and via-segment occupancy."""
        total = 0
        for use_name, cap_name in (
            ("via_col_use", "via_col_cap"),
            ("via_seg_use", "via_seg_cap"),
        ):
            if not hasattr(self, use_name) or not hasattr(self, cap_name):
                continue
            use = getattr(self, use_name)
            cap = getattr(self, cap_name)
            xp = (
                cp
                if GPU_AVAILABLE and isinstance(use, cp.ndarray)
                else np
            )
            subtotal = xp.maximum(0, use - cap).sum()
            if hasattr(subtotal, "get"):
                subtotal = subtotal.get()
            total += int(subtotal)
        return total

    def _block_via_edges_with_collisions(self):
        """
        Hard-block via edges with spatial collisions by setting costs to infinity.

        This prevents PathFinder from using via edges where the column or any
        spanned segment is already at capacity. Combined with soft penalties,
        this ensures no via spatial violations in the final routing.

        Uses GPU-accelerated CUDA kernel when available (30s → <1ms speedup!)
        """
        if not hasattr(self, '_via_edge_metadata') or self._via_edge_metadata is None:
            logger.warning("[HARD-BLOCK] No via edge metadata, skipping")
            return

        if not hasattr(self, 'via_col_use') and not hasattr(self, 'via_seg_use'):
            # No via spatial tracking enabled
            return

        # Try GPU kernel first if available
        if hasattr(self, 'via_kernel_manager') and self.via_kernel_manager.use_gpu:
            try:
                blocked_count = self.via_kernel_manager.hard_block_via_edges(
                    via_metadata=self._via_edge_metadata,
                    via_col_use_gpu=self.via_col_use,
                    via_col_cap_gpu=self.via_col_cap,
                    via_seg_use_gpu=self.via_seg_use if hasattr(self, 'via_seg_use') else None,
                    via_seg_cap_gpu=self.via_seg_cap if hasattr(self, 'via_seg_cap') else None,
                    total_cost_gpu=self.accounting.total_cost,
                    Ny=self._Ny,
                    segZ=self._segZ if hasattr(self, '_segZ') else 0
                )
                return  # GPU kernel succeeded
            except Exception as e:
                logger.warning(f"[HARD-BLOCK] GPU kernel failed: {e}, falling back to CPU")

        # CPU fallback
        via_edges = self._via_edge_metadata
        edge_indices = via_edges['indices']
        xy_coords = via_edges['xy_coords']
        z_lo = via_edges['z_lo']
        z_hi = via_edges['z_hi']

        # Convert from GPU to CPU if needed
        if hasattr(edge_indices, 'get'):
            edge_indices = edge_indices.get()
        if hasattr(xy_coords, 'get'):
            xy_coords = xy_coords.get()
        if hasattr(z_lo, 'get'):
            z_lo = z_lo.get()
        if hasattr(z_hi, 'get'):
            z_hi = z_hi.get()

        # Get cost array
        total_cost = self.accounting.total_cost
        if self.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Get via arrays (convert from GPU if needed)
        via_col_use = self.via_col_use.get() if hasattr(self.via_col_use, 'get') else self.via_col_use
        via_col_cap = self.via_col_cap.get() if hasattr(self.via_col_cap, 'get') else self.via_col_cap
        via_seg_use = self.via_seg_use.get() if hasattr(self, 'via_seg_use') and hasattr(self.via_seg_use, 'get') else getattr(self, 'via_seg_use', None)
        via_seg_cap = self.via_seg_cap.get() if hasattr(self, 'via_seg_cap') and hasattr(self.via_seg_cap, 'get') else getattr(self, 'via_seg_cap', None)

        blocked_count = 0

        for i in range(len(edge_indices)):
            xu, yu = int(xy_coords[i, 0]), int(xy_coords[i, 1])
            z_start, z_end = int(z_lo[i]), int(z_hi[i])
            edge_idx = edge_indices[i]

            # Check column capacity
            col_blocked = False
            if via_col_use is not None and via_col_cap is not None:
                if via_col_use[xu, yu] >= via_col_cap[xu, yu]:
                    col_blocked = True

            # Check segment capacity for all spanned segments
            seg_blocked = False
            if via_seg_use is not None and via_seg_cap is not None:
                for z in range(z_start, z_end):
                    seg_idx = z - 1  # Segments indexed from 0
                    if 0 <= seg_idx < self._segZ:
                        if via_seg_use[xu, yu, seg_idx] >= via_seg_cap[xu, yu, seg_idx]:
                            seg_blocked = True
                            break

            # Hard-block if at capacity
            if col_blocked or seg_blocked:
                total_cost_cpu[edge_idx] = np.float32('inf')
                blocked_count += 1

        # Copy back to GPU if needed
        if self.accounting.use_gpu:
            total_cost[:len(total_cost_cpu)] = cp.asarray(total_cost_cpu)

        if blocked_count > 0:
            logger.info(f"[HARD-BLOCK-CPU] Blocked {blocked_count} via edges at capacity (using CPU fallback)")

    def _pathfinder_negotiation(self, tasks: Dict[str, Tuple[int, int]], progress_cb=None, iteration_cb=None) -> Dict:
        """CORE PATHFINDER ALGORITHM WITH AUTO-CONFIGURATION"""
        cfg = self.config

        # ====================================================================
        # STEP 0: AUTO-CONFIGURE FROM BOARD CHARACTERISTICS
        # ====================================================================
        # Analyze board and derive optimal parameters (no manual tuning!)
        board_chars = analyze_board_characteristics(self.lattice, tasks)
        derived_params = derive_routing_parameters(board_chars)

        # Apply derived parameters to config (can be overridden by env vars)
        apply_derived_parameters(cfg, derived_params)

        # Store H/V layer assignments for later use
        self.h_layers = board_chars.h_layers
        self.v_layers = board_chars.v_layers

        # Load params into local variables with defaults (ensures new config values used)
        # Scale parameters by layer count for self-tuning
        n_sig_layers = self._Nz - 2  # Exclude F.Cu and B.Cu (layers 0 and Nz-1)

        # Base parameters from config
        pres_fac = float(getattr(cfg, 'pres_fac_init', 1.0))
        pres_fac_mult = float(getattr(cfg, 'pres_fac_mult', 1.15))
        hist_gain = float(getattr(cfg, 'hist_gain', 0.8))

        # Scale history by layer count (fewer layers need stronger memory).
        # Present pressure uses the configured ceiling with a layer-dependent
        # minimum; the previous code read cfg.pres_fac_max and then silently
        # discarded it.
        if n_sig_layers <= 12:
            hist_cost_weight_mult = 1.2  # 12.0 for few layers
        elif n_sig_layers <= 20:
            hist_cost_weight_mult = 1.0  # 10.0
        else:
            hist_cost_weight_mult = 0.8  # 8.0 for many layers
        pres_fac_max = resolve_pres_fac_max(cfg, n_sig_layers)

        # Allow env overrides for testing
        pres_fac_mult = float(os.getenv('ORTHO_PRES_FAC_MULT', pres_fac_mult))
        pres_fac_max = float(os.getenv('ORTHO_PRES_FAC_MAX', pres_fac_max))
        hist_gain = float(os.getenv('ORTHO_HIST_GAIN', hist_gain))
        history_decay = resolve_history_decay(cfg)

        best_overuse = float('inf')
        best_route_score = None
        best_route_state = None
        best_route_iteration = 0
        stagnant = 0
        prev_over_sum = float('inf')
        negotiated_overuse_history = []
        self._slow_progress_event_count = 0
        self._last_slow_progress_fraction = None
        self._hotset_rate_boost_until = 0
        self._last_slow_progress_trigger = -1_000_000
        self._pres_fac_max_now = pres_fac_max

        self._negotiation_ran = True

        logger.info(f"[NEGOTIATE] {len(tasks)} nets, {cfg.max_iterations} iters")
        logger.info(
            "[PARAMS] layers=%d pres_fac_init=%.2f pres_fac_mult=%.2f "
            "pres_fac_max=%.0f hist_gain=%.2f hist_weight=%.1f "
            "history_decay=%.3f",
            n_sig_layers,
            pres_fac,
            pres_fac_mult,
            pres_fac_max,
            hist_gain,
            cfg.hist_cost_weight * hist_cost_weight_mult,
            history_decay,
        )

        # Build edge→src mapping for barrel conflict detection
        logger.warning("[BARREL-CONFLICT-INIT] Building edge_src_map once before routing")
        self._ensure_edge_src_map()

        for it in range(1, cfg.max_iterations + 1):
            self.iteration = it
            self._pres_fac_now = pres_fac  # read by _build_owner_penalty
            self._pres_fac_max_now = pres_fac_max
            logger.info(f"[ITER {it}] pres_fac={pres_fac:.2f}")

            # Log iteration 1 always-connect policy
            if it == 1 and cfg.iter1_always_connect:
                logger.info("[ITER-1-POLICY] Always-connect mode: soft costs only (no hard blocks)")

            # STEP 0: Clean accounting rebuild (iter 2+)
            if it > 1 and cfg.reroute_only_offenders:
                # Rebuild usage from all currently routed nets before building hotset
                committed_nets = {nid for nid, path in self.net_paths.items() if path}
                self._rebuild_usage_from_committed_nets(committed_nets)
            else:
                # STEP 1: Refresh (iter 1 only, or if not using hotsets)
                self.accounting.refresh_from_canonical()

            # STEP 1.5: Rebuild via column/segment usage from committed paths
            self._rebuild_via_usage_from_committed()

            # STEP 1.6: Smooth via present costs and build segment prefix
            alpha = float(getattr(cfg, "via_present_alpha", 0.6))
            beta = float(getattr(cfg, "via_present_beta", 0.4))

            if hasattr(self, 'via_col_use'):
                over = np.maximum(0, self.via_col_use - self.via_col_cap).astype(np.float32)
                self.via_col_pres = alpha * over + beta * self.via_col_pres

            if hasattr(self, 'via_seg_use'):
                over = np.maximum(0, self.via_seg_use - self.via_seg_cap).astype(np.float32)
                self.via_seg_pres = alpha * over + beta * self.via_seg_pres
                # Build cumulative prefix along z axis for fast range queries
                np.cumsum(self.via_seg_pres, axis=2, out=self.via_seg_prefix)

            # OPTIMIZATION: Cache GPU arrays to avoid redundant transfers (200MB × 4-5 times = slow!)
            # This saves 2-4 seconds per iteration by transferring once instead of 4-5 times
            if self.accounting.use_gpu:
                present_cpu_cache = self.accounting.present.get()
                cap_cpu_cache = self.accounting.capacity.get()
            else:
                present_cpu_cache = self.accounting.present
                cap_cpu_cache = self.accounting.capacity

            # STEP 1.7: Update present EMA for stable convergence
            # Initialize present_ema on first iteration
            if it == 1:
                self.accounting.present_ema = self.accounting.present.copy()
            # Heavy smoothing for stability (40% new, 60% old)
            present_ema_beta = float(getattr(cfg, 'present_ema_beta', 0.40))
            self.accounting.update_present_ema(beta=present_ema_beta)

            # STEP 1.8: Compute layer bias for layer balancing
            layer_bias = self._compute_layer_bias(
                self.accounting, self.graph,
                num_layers=self.lattice.layers,
                alpha=0.88,  # Balanced smoothing
                max_boost=1.80  # Conservative penalty (baseline from document)
            )

            # STEP 2: Update costs (with history weight and via annealing)
            # Via policy: anneal via cost when pres_fac >= 64 (lowered to trigger earlier)
            via_cost_mult = 1.0
            spatial_via_overuse = self._spatial_via_overuse_total()
            via_pressure_threshold = int(getattr(
                cfg, "via_pressure_threshold", 64
            ))
            if (
                it >= 3
                and spatial_via_overuse >= via_pressure_threshold
            ):
                via_cost_mult = float(getattr(
                    cfg, "via_pressure_multiplier", 1.5
                ))
                logger.info(
                    "[VIA-PRESSURE] Spatial overuse=%d >= %d; "
                    "via base cost *= %.2f",
                    spatial_via_overuse,
                    via_pressure_threshold,
                    via_cost_mult,
                )
            elif pres_fac >= 64:
                # Check if >70% of overuse is on vias
                # OPTIMIZATION: Use cached GPU transfers
                present = present_cpu_cache
                cap = cap_cpu_cache
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

            # Get edge_layer and edge_kind for layer balancing
            edge_layer_arr = (self.graph.edge_layer_gpu if self.accounting.use_gpu
                            else self.graph.edge_layer) if hasattr(self.graph, 'edge_layer') else None
            edge_kind_arr = (self.graph.edge_kind_gpu if self.accounting.use_gpu
                           else self.graph.edge_kind) if hasattr(self.graph, 'edge_kind') else None

            # CRITICAL: Cost update ONCE per iteration (PathFinder design)
            # Toggle incremental updates via env var INCREMENTAL_COST_UPDATE=1
            active_hist_weight = cfg.hist_cost_weight
            if os.getenv("INCREMENTAL_COST_UPDATE") == "1" and hasattr(self, '_changed_edges_previous_iteration'):
                # INCREMENTAL: Only update edges that changed in previous iteration
                changed_edges = self._changed_edges_previous_iteration
                if hasattr(self.accounting, 'update_costs_incremental'):
                    self.accounting.update_costs_incremental(
                        changed_edges,
                        self.graph.base_costs, pres_fac,
                        hist_weight=cfg.hist_cost_weight,
                        via_cost_multiplier=via_cost_mult,
                        base_cost_weight=cfg.base_cost_weight,
                        edge_layer=edge_layer_arr,
                        layer_bias_per_layer=layer_bias,
                        edge_kind=edge_kind_arr
                    )
                    logger.debug(f"[ITER {it}] Incremental cost update: {len(changed_edges)} edges")
                else:
                    # Fallback if incremental not implemented
                    self.accounting.update_costs(
                        self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
                        via_cost_multiplier=via_cost_mult,
                        base_cost_weight=cfg.base_cost_weight,
                        edge_layer=edge_layer_arr,
                        layer_bias_per_layer=layer_bias,
                        edge_kind=edge_kind_arr
                    )
            else:
                # FULL: Update all edges (default PathFinder behavior) - use hist_cost_weight with layer scaling
                hist_weight_scaled = cfg.hist_cost_weight * hist_cost_weight_mult
                active_hist_weight = hist_weight_scaled
                self.accounting.update_costs(
                    self.graph.base_costs, pres_fac, hist_weight_scaled,
                    via_cost_multiplier=via_cost_mult,
                    base_cost_weight=cfg.base_cost_weight,
                    edge_layer=edge_layer_arr,
                    layer_bias_per_layer=layer_bias,
                    edge_kind=edge_kind_arr
                )

            # STEP 2.5: Apply via column/segment pooling penalties
            # NOTE: Layer balancing now enabled (vectorized, applied in update_costs)
            self._apply_via_pooling_penalties(pres_fac)

            # STEP 2.6: Hard-block via edges with spatial collisions (iteration 2+)
            # This prevents PathFinder from using edges that would cause via collisions
            if it > 1:  # Allow greedy first pass without hard blocking
                self._block_via_edges_with_collisions()

            # STEP 2.9: Optionally quarantine persistently failing nets for
            # diagnostics. Production routing keeps every net negotiable.
            if not hasattr(self, '_net_failure_count'):
                self._net_failure_count = {}  # net_id -> consecutive failures
                self._excluded_nets = set()

            allow_exclusion = bool(getattr(
                cfg, "allow_net_exclusion", False
            ))
            if (
                allow_exclusion
                and it > 10
                and it % 10 == 0
                and self._excluded_nets
            ):
                num_to_retry = len(self._excluded_nets)
                logger.warning(f"[EXCLUDE-RETRY] Iteration {it}: Giving {num_to_retry} excluded nets another chance")
                self._excluded_nets.clear()
                self._net_failure_count.clear()

            if allow_exclusion and it > 4:
                # Update failure counts based on which nets have paths
                for net_id in tasks.keys():
                    has_path = bool(self.net_paths.get(net_id))
                    if not has_path:
                        # Net failed to route
                        self._net_failure_count[net_id] = self._net_failure_count.get(net_id, 0) + 1
                    else:
                        # Net successfully routed - reset failure count
                        self._net_failure_count[net_id] = 0

                # Exclude nets that have failed 5+ consecutive times
                FAILURE_THRESHOLD = 5
                newly_excluded = set()
                for net_id, failures in self._net_failure_count.items():
                    if failures >= FAILURE_THRESHOLD and net_id not in self._excluded_nets:
                        self._excluded_nets.add(net_id)
                        newly_excluded.add(net_id)

                if newly_excluded:
                    logger.warning(f"[EXCLUDE] Giving up on {len(newly_excluded)} nets after {FAILURE_THRESHOLD} failed attempts: {list(newly_excluded)[:5]}...")

                if self._excluded_nets:
                    logger.info(f"[EXCLUDE] {len(self._excluded_nets)} nets permanently excluded from routing")

            # STEP 3: Route (hotset incremental after iter 1)
            # DIAGNOSTIC OVERRIDE: Test with hotset disabled
            FORCE_ROUTE_ALL = False  # Set to False to restore hotset behavior (TESTING: hotset ENABLED)

            if FORCE_ROUTE_ALL:
                sub_tasks = {k: v for k, v in tasks.items() if k not in self._excluded_nets}
                logger.info(f"  [DIAGNOSTIC] Routing ALL {len(sub_tasks)} nets every iteration (hotset DISABLED for testing)")
            elif cfg.reroute_only_offenders and it > 1:
                # Pass ripped set to _build_hotset (Fix 2)
                offenders = self._build_hotset(tasks, ripped=getattr(self, "_last_ripped", set()))
                # Exclude permanently failed nets from offenders
                offenders = offenders - self._excluded_nets
                sub_tasks = {k: v for k, v in tasks.items() if k in offenders and k not in self._excluded_nets}
                logger.info(f"  Hotset: {len(offenders)}/{len(tasks)} nets")
                # Clear _last_ripped after use
                self._last_ripped = set()
            else:
                sub_tasks = {k: v for k, v in tasks.items() if k not in self._excluded_nets}

            # ANTI-OSCILLATION: Shuffle net order each iteration to break deterministic patterns
            # This prevents the same nets from always winning/losing in the same order
            if it > 1:
                import random
                net_ids = list(sub_tasks.keys())
                random.seed(42 + it)  # Deterministic but different each iteration
                random.shuffle(net_ids)
                sub_tasks = {nid: sub_tasks[nid] for nid in net_ids}
                logger.debug(f"[SHUFFLE] Randomized net order for iteration {it}")

            pressure_work_scale = (
                self._pressure_work_scale(
                    len(sub_tasks),
                    reference_hotset=int(getattr(
                        cfg,
                        "pressure_reference_hotset",
                        100,
                    )),
                    maximum_scale=float(getattr(
                        cfg,
                        "pressure_work_scale_max",
                        2.0,
                    )),
                )
                if it > 1 else 1.0
            )
            self._last_pressure_work_scale = pressure_work_scale
            routed, failed = self._route_all(sub_tasks, all_tasks=tasks, pres_fac=pres_fac, iteration=it)

            # Track changed edges for next iteration's incremental cost update
            # Collect all edges used by nets routed in this iteration
            changed_edges = set()
            for net_id in sub_tasks.keys():
                if net_id in self._net_to_edges:
                    changed_edges.update(self._net_to_edges[net_id])
            # Store for next iteration (will be used by incremental cost update)
            self._changed_edges_previous_iteration = changed_edges

            # CRITICAL: Refresh present to reflect committed paths
            self.accounting.refresh_from_canonical()

            # ACCOUNTING SANITY CHECK: Verify present matches canonical
            if not self.accounting.verify_present_matches_canonical():
                logger.warning(f"[ITER {it}] Accounting mismatch detected - potential bug")

            # The pre-route host snapshot is valid for constructing this
            # iteration's costs, but not for post-route diagnostics or the
            # next layer-bias update. NumPy observed in-place changes while
            # CuPy's .get() returned a stale copy, so GPU policy lagged CPU by
            # one iteration.
            if self.accounting.use_gpu:
                post_present_cpu = self.accounting.present.get()
                post_cap_cpu = self.accounting.capacity.get()
            else:
                post_present_cpu = self.accounting.present
                post_cap_cpu = self.accounting.capacity

            # STEP 3.5: Detect via-barrel conflicts independently from edge
            # capacity. Barrel collisions identify nets to reroute; they are
            # not extra edge occupancy and must never be written into
            # accounting.present (or learned permanently by edge history).
            _, conflict_count = self._detect_barrel_conflicts()

            # Store barrel conflict count for iteration summary
            self._last_barrel_conflict_count = conflict_count
            # Graph congestion and exact physical conflicts require different
            # cleanup schedules. Measure every negotiated graph resource before
            # deciding whether physical offenders may enter the next hotset:
            # mixing terminal cleanup into a route with shared path nodes can
            # repair barrels by recreating graph collisions.
            over_sum, over_cnt = self.accounting.compute_overuse(
                router_instance=self
            )
            (
                path_node_overuse,
                path_node_overuse_count,
            ) = self._compute_path_node_overuse()
            self._last_path_node_overuse_total = path_node_overuse
            self._last_path_node_overuse_count = (
                path_node_overuse_count
            )
            negotiated_overuse = over_sum + path_node_overuse
            physical_cleanup_threshold = int(getattr(
                cfg,
                "portal_cleanup_edge_threshold",
                3,
            ))
            physical_cleanup_ready = (
                over_cnt <= physical_cleanup_threshold
                and negotiated_overuse <= physical_cleanup_threshold
            )
            for portal_key in getattr(
                self, "_barrel_owner_portal_keys", ()
            ):
                self._portal_barrel_history[portal_key] += 1.0

            physical_conflicts = int(conflict_count)
            if physical_conflicts > 0:
                previous = getattr(
                    self, "_escape_replan_best", float("inf")
                )
                if physical_conflicts < previous:
                    self._escape_replan_best = physical_conflicts
                    self._escape_replan_stagnant = 0
                else:
                    self._escape_replan_stagnant = (
                        getattr(self, "_escape_replan_stagnant", 0) + 1
                    )
                replans = getattr(self, "_escape_replan_count", 0)
                if (
                    int(getattr(
                        self, "_last_escape_conflict_count", 0
                    )) > 0
                    and physical_cleanup_ready
                    and not getattr(
                        self, "_freeze_selected_portals", False
                    )
                    and
                    self._escape_replan_stagnant
                    >= int(getattr(
                        cfg, "escape_replan_patience", 2
                    ))
                    and replans
                    < int(getattr(cfg, "escape_replan_limit", 3))
                ):
                    logger.warning(
                        "[ESCAPE-REPLAN] Re-solving %d stalled "
                        "physical conflicts",
                        physical_conflicts,
                    )
                    self._plan_escape_assignment()
                    self._escape_replan_count = replans + 1
                    self._escape_replan_stagnant = 0
            else:
                self._escape_replan_best = float("inf")
                self._escape_replan_stagnant = 0

            if conflict_count > 0 and physical_cleanup_ready:
                logger.debug(f"[BARREL-CONFLICT] Detected {conflict_count} barrel conflicts in iteration {it}")
                self._last_ripped = (
                    set(getattr(self, "_last_ripped", ()))
                    | set(getattr(self, "_barrel_conflict_nets", ()))
                )

            # STEP 4: Edge overuse. Barrel conflicts remain a separate
            # convergence condition and hotset input.

            # Instrumentation: via overuse ratio
            present = post_present_cpu
            cap = post_cap_cpu
            over = np.maximum(0, present - cap)
            # Use numpy boolean indexing for efficient via overuse calculation
            via_overuse = float(over[self._via_edges[:len(over)]].sum())
            via_ratio = (via_overuse / over_sum * 100) if over_sum > 0 else 0.0

            # Check barrel conflicts
            barrel_conflicts = getattr(self, '_last_barrel_conflict_count', 0)
            exact_barrel_conflicts = getattr(
                self, "_last_exact_barrel_conflict_count", 0
            )
            path_node_conflicts = getattr(
                self, "_last_path_node_conflict_count", 0
            )
            escape_conflicts = getattr(
                self, "_last_escape_conflict_count", 0
            )
            portal_grid_conflicts = getattr(
                self, "_last_portal_grid_conflict_count", 0
            )

            # Once every negotiated graph resource is clean, repair physical
            # conflicts one side at a time while a tiny complete-resource tail
            # keeps negotiating. Waiting for physical conflicts themselves to
            # vanish would let both ends of a short move together.
            if self._should_run_one_sided_cleanup(
                barrel_conflicts,
                over_cnt,
                getattr(self, "_freeze_selected_portals", False),
                int(getattr(
                    cfg,
                    "portal_cleanup_edge_threshold",
                    3,
                )),
                overuse_total=negotiated_overuse,
            ):
                entering_cleanup = not getattr(
                    self, "_freeze_selected_portals", False
                )
                if entering_cleanup:
                    self._physical_cleanup_started = True
                    logger.info(
                        "[PORTAL-CLEANUP] Freezing %d selected terminal "
                        "vias; rerouting %d grid victims",
                        sum(
                            len(portals)
                            for portals in
                            self.net_selected_portals.values()
                        ),
                        len(getattr(
                            self, "_portal_grid_victim_nets", ()
                        )),
                    )
                self._freeze_selected_portals = True
                self._portal_cleanup_movable_nets = (
                    self._portal_cleanup_movable_components(
                        getattr(self, "_portal_grid_pairs", ()),
                        getattr(self, "_escape_conflict_pairs", ()),
                        (
                            set(getattr(
                                self, "_exact_barrel_pairs", ()
                            ))
                            | set(getattr(
                                self,
                                "_path_node_conflict_pairs",
                                (),
                            ))
                        ),
                    )
                )
                cleanup_involved_nets = (
                    set(getattr(
                        self, "_portal_grid_owner_nets", ()
                    ))
                    | set(getattr(
                        self, "_portal_grid_victim_nets", ()
                    ))
                    | {
                        key[0]
                        for pair in getattr(
                            self, "_escape_conflict_pairs", ()
                        )
                        for key in pair
                    }
                )
                logger.info(
                    "[PORTAL-CLEANUP] Holding %d involved nets fixed; "
                    "allowing %d conflict peers to renegotiate punch-ins",
                    len(
                        cleanup_involved_nets
                        - self._portal_cleanup_movable_nets
                    ),
                    len(self._portal_cleanup_movable_nets),
                )
                # Movable terminals may have changed during the preceding
                # iteration. Re-index the current exact envelopes before the
                # next hotset routes; live node ownership covers changes made
                # later within that iteration.
                self._rebuild_portal_cleanup_edge_owners()
                if self._portal_cleanup_movable_nets:
                    self._barrel_conflict_nets = set(
                        self._portal_cleanup_movable_nets
                    )
                    self._last_ripped = set(
                        self._portal_cleanup_movable_nets
                    )
            elif barrel_conflicts == 0:
                self._freeze_selected_portals = False
                self._portal_cleanup_movable_nets = set()
                self._portal_cleanup_move_counts.clear()
            elif (
                getattr(self, "_freeze_selected_portals", False)
                and (
                    over_cnt > int(getattr(
                        cfg,
                        "portal_cleanup_edge_threshold",
                        3,
                    ))
                    or negotiated_overuse > int(getattr(
                        cfg,
                        "portal_cleanup_edge_threshold",
                        3,
                    ))
                )
            ):
                logger.info(
                    "[PORTAL-CLEANUP] Pausing physical cleanup after "
                    "graph overuse reopened to %d edges",
                    over_cnt,
                )
                self._freeze_selected_portals = False
                self._portal_cleanup_movable_nets = set()
                self._portal_cleanup_edge_members = {}
                self._portal_cleanup_foreign_cache = {}
            elif (
                escape_conflicts > 0
                and not getattr(
                    self, "_freeze_selected_portals", False
                )
            ):
                self._freeze_selected_portals = False
                self._portal_cleanup_movable_nets = set()

            # Clean consolidated iteration summary (WARNING level so it shows in console)
            status = (
                "✓ CONVERGED"
                if negotiated_overuse == 0
                else (
                    f"overuse={over_sum}"
                    f" node_overuse={path_node_overuse}"
                    f" negotiated={negotiated_overuse}"
                )
            )
            barrel_info = (
                f"  barrel={barrel_conflicts}"
                f" (exact={exact_barrel_conflicts},"
                f" node={path_node_conflicts},"
                f" escape={escape_conflicts})"
                if barrel_conflicts > 0 else ""
            )
            logger.warning(f"[ITER {it:3d}] nets={routed}/{routed+failed}  {status}  edges={over_cnt}  via_overuse={via_ratio:.0f}%{barrel_info}")

            # Retain the best complete routing state, not just its scalar
            # metric. Negotiation deliberately oscillates, so the final
            # iteration can be worse than an earlier pass. Rank states
            # lexicographically: route every net first, then minimize every
            # negotiated graph resource (edges, via pools, and capacity-one
            # nodes), then minimize off-graph physical conflicts.
            route_score = self._negotiated_route_score(
                failed,
                over_sum,
                path_node_overuse,
                conflict_count,
            )
            if best_route_score is None or route_score < best_route_score:
                best_route_score = route_score
                best_route_state = self._capture_routing_state()
                best_route_iteration = it
                # A better basin deserves a fresh bounded-recovery search.
                # Within one retained basin, recovery waves rotate through
                # untried high-impact victims instead of deterministically
                # ripping the same top-k nets after every rollback.
                self._stagnation_victim_history = set()

            # A monster route that improves by only a few units per pass is
            # operationally plateaued even though the strict best-value
            # counter resets. Detect inadequate rolling descent, widen the
            # next several severe hotsets, then raise the pressure ceiling
            # only after the wider waves have also had a full window.
            negotiated_overuse_history.append(negotiated_overuse)
            slow_window = max(
                1,
                int(getattr(cfg, "slow_progress_window", 5)),
            )
            slow_progress, progress_fraction = (
                self._rolling_progress_insufficient(
                    negotiated_overuse_history,
                    window=slow_window,
                    minimum_fraction=float(getattr(
                        cfg,
                        "slow_progress_min_fraction",
                        0.025,
                    )),
                    minimum_overuse=int(getattr(
                        cfg,
                        "slow_progress_min_overuse",
                        16_384,
                    )),
                )
            )
            self._last_slow_progress_fraction = progress_fraction
            trigger_separated = (
                it - self._last_slow_progress_trigger >= slow_window
            )
            if slow_progress and trigger_separated:
                self._last_slow_progress_trigger = it
                self._slow_progress_event_count += 1
                self._hotset_rate_boost_until = it + slow_window
                pressure_after = max(
                    1,
                    int(getattr(
                        cfg,
                        "slow_progress_pressure_after",
                        2,
                    )),
                )
                old_ceiling = pres_fac_max
                if self._slow_progress_event_count >= pressure_after:
                    ultimate_ceiling = max(
                        old_ceiling,
                        float(getattr(
                            cfg,
                            "slow_progress_pres_fac_max",
                            256.0,
                        )),
                    )
                    pres_fac_max = min(
                        ultimate_ceiling,
                        max(128.0, old_ceiling * 2.0),
                    )
                    pres_fac = min(
                        pres_fac * 1.5,
                        pres_fac_max,
                    )
                self._pres_fac_max_now = pres_fac_max
                boosted_hotset_cap = self._effective_history_hotset_cap(
                    negotiated_overuse
                )
                logger.warning(
                    "[RATE-PLATEAU %d] Best rolling descent over %d "
                    "passes is %.3f%%; widening severe hotsets to %d "
                    "through iteration %d and pressure ceiling %.0f -> %.0f",
                    self._slow_progress_event_count,
                    slow_window,
                    100.0 * progress_fraction,
                    boosted_hotset_cap,
                    self._hotset_rate_boost_until,
                    old_ceiling,
                    pres_fac_max,
                )

            # DIAGNOSTIC: Verify history is growing (not capped at 1.0) - only first 3 iterations
            if it <= 3:
                hist_max = float(self.accounting.history.max())
                if hist_max <= 1.1:
                    logger.warning(f"[HISTORY] Iter {it}: hist_max={hist_max:.1f} (still capped?)")
                else:
                    logger.debug(f"[HISTORY] Iter {it}: hist_max={hist_max:.1f} (growing)")

            # Convergence diagnostics (every 10 iterations, reduced from every 5)
            if it % 10 == 0:
                cost_ratio = self.accounting.cost_balance_ratio(
                    active_hist_weight,
                    pres_fac,
                )

                logger.debug(f"[CONVERGENCE] pres_fac={pres_fac:.2f} hist_gain={hist_gain:.2f} balance={cost_ratio:.2f}")

                if cost_ratio < 0.1 or cost_ratio > 10.0:
                    logger.warning(f"[CONVERGENCE] Cost imbalance: ratio={cost_ratio:.3f} (target: 0.5-2.0)")

            # INSTRUMENTATION: Report hard wall count for iter-1 (must be 0)
            if it == 1 and hasattr(self, '_iter1_inf_writes'):
                inf_total = self._iter1_inf_writes
                if inf_total == 0:
                    logger.info(f"[ITER-1-HARDWALLS] ✓ count=0 (no infinite costs in iteration 1)")
                else:
                    logger.error(f"[ITER-1-HARDWALLS] ✗ count={inf_total} (BUG: infinite costs found in iteration 1!)")
                self._iter1_inf_writes = 0  # Reset for next test

            # Instrumentation: Via pooling statistics
            # OPTIMIZATION: Reduced frequency to every 10 iterations (0.5-1s speedup)
            if it % 10 == 0 and (hasattr(self, 'via_col_use') or hasattr(self, 'via_seg_use')):
                if hasattr(self, 'via_col_use'):
                    cols_used = np.sum(self.via_col_use > 0)
                    cols_over = np.sum(self.via_col_use > self.via_col_cap)
                    max_col_use = int(np.max(self.via_col_use))
                    max_col_pres = float(np.max(self.via_col_pres))
                    mean_col_pres = float(np.mean(self.via_col_pres[self.via_col_pres > 0])) if np.any(self.via_col_pres > 0) else 0.0
                    logger.info(f"[VIA-POOL] Columns: used={cols_used}, over_cap={cols_over}, max_use={max_col_use}, max_pres={max_col_pres:.2f}, mean_pres={mean_col_pres:.2f}")

                if hasattr(self, 'via_seg_use'):
                    segs_used = np.sum(self.via_seg_use > 0)
                    segs_over = np.sum(self.via_seg_use > self.via_seg_cap)
                    max_seg_use = int(np.max(self.via_seg_use))
                    max_seg_pres = float(np.max(self.via_seg_pres))
                    max_seg_prefix = float(np.max(self.via_seg_prefix))
                    logger.info(f"[VIA-POOL] Segments: used={segs_used}, over_cap={segs_over}, max_use={max_seg_use}, max_pres={max_seg_pres:.2f}, max_prefix={max_seg_prefix:.2f}")

            # Instrumentation: Per-layer congestion breakdown (GPU-vectorized - fast!)
            overuse_by_layer = None
            if over_sum > 0:
                overuse_by_layer = self._log_per_layer_congestion(over)

            # Top-10 channels only every 10 iters (still expensive due to coordinate conversion)
            if over_sum > 0 and it % 10 == 0:
                self._log_top_overused_channels(over, top_k=10)

            # Update layer bias from horizontal overuse (EWMA for stability)
            if hasattr(self, 'layer_bias') and hasattr(self.graph, 'edge_layer'):
                if overuse_by_layer is None and over_sum > 0:
                    overuse_by_layer = self._log_per_layer_congestion(over)

                if overuse_by_layer and sum(overuse_by_layer.values()) > 0:
                    # Compute pressure per layer (normalized to mean)
                    layer_overuse = np.array([overuse_by_layer.get(z, 0.0) for z in range(self._Nz)])
                    mean_overuse = np.mean(layer_overuse[layer_overuse > 0]) + 1e-9
                    # Pressure ratio: >1.0 = hotter than average, <1.0 = cooler
                    pressure = layer_overuse / mean_overuse

                    # Target bias (1.0 = neutral, <1.0 = cheaper, >1.0 = more expensive)
                    # Scale alpha by layer count: fewer layers need stronger balancing
                    n_sig_layers = self._Nz - 2  # Exclude F.Cu and B.Cu
                    if n_sig_layers <= 12:
                        alpha = 0.20  # Strong balancing for sparse stacks
                        bias_min, bias_max = 0.60, 1.80
                    elif n_sig_layers <= 20:
                        alpha = 0.12
                        bias_min, bias_max = 0.75, 1.50
                    else:
                        alpha = 0.08
                        bias_min, bias_max = 0.85, 1.20

                    target_bias = 1.0 + alpha * (pressure - 1.0)

                    # EWMA smoothing
                    self.layer_bias = 0.85 * self.layer_bias + 0.15 * target_bias
                    self.layer_bias = np.clip(self.layer_bias, bias_min, bias_max)

                    # Log top biases
                    top_layers = sorted(enumerate(self.layer_bias), key=lambda x: x[1], reverse=True)[:3]
                    if any(abs(bias - 1.0) > 0.03 for _, bias in top_layers):
                        logger.info(f"[LAYER-BIAS] Hot layers: " +
                                   ", ".join([f"L{z}:{bias:.3f}" for z, bias in top_layers if 1 <= z < self._Nz-1]))

                    # Layer jam breaker: Detect if one layer dominates overuse
                    total_layer_overuse = sum(overuse_by_layer.values())
                    if total_layer_overuse > 0:
                        # Find layer with highest overuse percentage
                        max_layer = max(overuse_by_layer.items(), key=lambda x: x[1])
                        max_layer_idx, max_layer_overuse = max_layer
                        jam_percentage = (max_layer_overuse / total_layer_overuse) * 100

                        # Track consecutive jams
                        if not hasattr(self, '_layer_jam_tracker'):
                            self._layer_jam_tracker = {'layer': None, 'count': 0, 'boost_until': 0}

                        if jam_percentage >= 60.0:
                            if self._layer_jam_tracker['layer'] == max_layer_idx:
                                self._layer_jam_tracker['count'] += 1
                            else:
                                self._layer_jam_tracker = {'layer': max_layer_idx, 'count': 1, 'boost_until': 0}

                            # If jammed for 3+ iterations, temporarily boost that layer's bias
                            if self._layer_jam_tracker['count'] >= 3 and it > self._layer_jam_tracker['boost_until']:
                                # Boost toward bias_max to make layer more expensive
                                # Since bias < 1.0 means cheaper, we need to increase it toward bias_max
                                old_bias = self.layer_bias[max_layer_idx]
                                self.layer_bias[max_layer_idx] = min(old_bias * 1.8, bias_max)  # Stronger boost
                                self._layer_jam_tracker['boost_until'] = it + 3  # Hold longer
                                logger.warning(f"[LAYER-JAM] Layer {max_layer_idx} jammed at {jam_percentage:.1f}% for {self._layer_jam_tracker['count']} iters → boosting bias {old_bias:.3f} → {self.layer_bias[max_layer_idx]:.3f} for 3 iters")
                        else:
                            # Reset if no longer jammed
                            self._layer_jam_tracker = {'layer': None, 'count': 0, 'boost_until': 0}

            # Clean-phase: if overuse==0, freeze good nets and finish stragglers
            if over_sum == 0:
                unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}
                if not unrouted:
                    logger.info(
                        "[CLEAN] All nets routed with zero edge overuse; "
                        "continuing to the barrel audit"
                    )
                # Freeze placed nets and lower pressure for stragglers
                placed = {nid for nid in tasks.keys() if self.net_paths.get(nid)}
                pres_fac = min(pres_fac, 10.0)
                logger.info(f"[CLEAN] Overuse=0, {len(unrouted)} unrouted left → freeze {len(placed)} nets, pres_fac={pres_fac:.2f}")

            # STEP 5: History (use local hist_gain variable, optionally boost after iter 8)
            hist_gain_eff = hist_gain
            if it >= 8 and over_sum > 0:
                # Strengthen history memory after learning phase
                hist_gain_eff = min(1.2, hist_gain * 1.25)

            # TEST: Allow env override to disable history cap and use raw present
            use_history_cap = os.getenv('ORTHO_NO_HISTORY_CAP', '0') == '0'
            hist_base_costs = self.graph.base_costs if use_history_cap else None
            hist_cap_mult = 15.0 if use_history_cap else 1.0  # Reduced from 100.0 to prevent history dominance
            # FIX: Use raw present for history (present_ema lags significantly in early iterations)
            use_raw_present = True  # was: os.getenv('ORTHO_RAW_PRESENT_FOR_HIST', '0') == '1'

            self.accounting.update_history(
                hist_gain_eff,
                base_costs=hist_base_costs,
                history_cap_multiplier=hist_cap_mult,
                decay_factor=history_decay,
                use_raw_present=use_raw_present
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

                    # CRITICAL: Merge escape geometry for complete board state visualization
                    # Without this, iteration screenshots only show routed nets, not pad escapes
                    if hasattr(self, '_escape_tracks') and self._escape_tracks:
                        # Deduplicate helper
                        def _dedupe(items, key_fn):
                            seen, out = set(), []
                            for item in items:
                                k = key_fn(item)
                                if k not in seen:
                                    seen.add(k)
                                    out.append(item)
                            return out

                        # Merge escapes + routed geometry
                        combined_tracks = self._escape_tracks + provisional_tracks
                        combined_vias = self._escape_vias + provisional_vias

                        # Deduplicate by geometric signature (safe key access to avoid None subscripting)
                        def safe_track_key(t):
                            start = t.get("start") or t.get("x1", 0), t.get("y1", 0)
                            end = t.get("end") or t.get("x2", 0), t.get("y2", 0)
                            return (t.get("net"), t.get("layer"),
                                   round(start[0] if isinstance(start, (list, tuple)) else 0, 3),
                                   round(start[1] if isinstance(start, (list, tuple)) else 0, 3),
                                   round(end[0] if isinstance(end, (list, tuple)) else 0, 3),
                                   round(end[1] if isinstance(end, (list, tuple)) else 0, 3),
                                   round(t.get("width", 0), 3))

                        def safe_via_key(v):
                            at_pos = v.get("at") or v.get("x", 0), v.get("y", 0)
                            return (v.get("net"),
                                   round(at_pos[0] if isinstance(at_pos, (list, tuple)) else 0, 3),
                                   round(at_pos[1] if isinstance(at_pos, (list, tuple)) else 0, 3),
                                   v.get("layers") or (v.get("from_layer"), v.get("to_layer")),
                                   round(v.get("size", 0), 3),
                                   round(v.get("drill", 0), 3),
                                   round(v.get("diameter", 0), 3))

                        provisional_tracks = _dedupe(combined_tracks, safe_track_key)
                        provisional_vias = _dedupe(combined_vias, safe_via_key)

                        logger.debug(f"[ITER {it}] Screenshot: escapes={len(self._escape_tracks)} + "
                                    f"routed={len(provisional_tracks) - len(self._escape_tracks)} → "
                                    f"total={len(provisional_tracks)} tracks, {len(provisional_vias)} vias")

                    iteration_cb(it, provisional_tracks, provisional_vias, over_sum, over_cnt)
                except Exception as e:
                    logger.warning(f"[ITER {it}] Iteration callback failed: {e}")

            # STEP 6: Terminate?
            if failed == 0 and over_sum == 0:
                # A zero edge-overuse state is rare and worth a fresh audit;
                # never trust a skipped/stale barrel count for export.
                self._rebuild_node_owner()
                _, barrel_conflicts = self._detect_barrel_conflicts()
                self._last_barrel_conflict_count = barrel_conflicts
                if barrel_conflicts == 0:
                    logger.info("[SUCCESS] Zero overuse, zero failed nets, AND zero barrel conflicts!")
                else:
                    logger.info(f"[SUCCESS] Zero overuse and zero failed nets ({barrel_conflicts} barrel conflicts remain)")

                # Final collision detection validation
                edges_over_capacity = [(e, usage) for e, usage in self.accounting.edge_usage.items() if usage > 1]
                if edges_over_capacity:
                    logger.error(f"[COLLISION] {len(edges_over_capacity)} edges over capacity (SHOULD BE ZERO!)")
                    for e, usage in edges_over_capacity[:10]:  # Show first 10
                        logger.error(f"[COLLISION]   Edge {e}: {usage} nets (capacity=1)")
                else:
                    logger.info("[COLLISION] 0 edges over capacity ✓ PERFECT!")

                # Check barrel conflicts before declaring full convergence
                if barrel_conflicts > 0:
                    logger.warning(f"[CONVERGENCE] Edge overuse=0 but {barrel_conflicts} barrel conflicts remain")
                    logger.warning(f"[CONVERGENCE] Continuing to iteration {it+1} to resolve barrel conflicts...")
                    self._last_ripped = set(
                        getattr(self, "_barrel_conflict_nets", ())
                    )
                else:
                    # TRUE convergence: zero edge overuse AND zero barrel conflicts
                    logger.info("[COLLISION] 0 edges over capacity ✓ PERFECT!")

                    # Log GPU vs CPU pathfinding statistics
                    total_paths = self._gpu_path_count + self._cpu_path_count
                    if total_paths > 0:
                        gpu_pct = (self._gpu_path_count / total_paths) * 100
                        logger.info(f"[GPU-STATS] GPU: {self._gpu_path_count} paths ({gpu_pct:.1f}%), CPU: {self._cpu_path_count} paths ({100-gpu_pct:.1f}%)")

                    return {'success': True, 'paths': self.net_paths, 'converged': True}

            if negotiated_overuse < best_overuse:
                best_overuse = negotiated_overuse
                stagnant = 0
            else:
                stagnant += 1

            # Plateau kicker: If stuck for 2 iterations, boost present pressure
            if stagnant == 2:
                old_pres_fac = pres_fac
                pres_fac = min(pres_fac * 1.5, pres_fac_max)
                logger.info(f"[PLATEAU-KICK] Stagnant for 2 iters, boosting pres_fac: {old_pres_fac:.2f} → {pres_fac:.2f}")

            if (
                stagnant >= cfg.stagnation_patience
                and it < cfg.max_iterations
            ):
                spatial_via_overuse = (
                    self._spatial_via_overuse_total()
                )
                via_tail_threshold = (
                    self._via_keeper_rotation_threshold()
                )
                if not self._should_rip_for_stagnation(
                    spatial_via_overuse,
                    via_tail_threshold,
                    physical_cleanup_started=getattr(
                        self, "_physical_cleanup_started", False
                    ),
                ):
                    # Via-pool offenders are already selected directly by
                    # _build_hotset. Ripping additional ordinary-edge nets
                    # during broad via recovery replaces otherwise-good full
                    # paths and can recreate hundreds of spatial collisions.
                    logger.info(
                        "[STAGNATION] Suppressing speculative rip-up while "
                        "spatial-via overuse=%d exceeds tail threshold=%d",
                        spatial_via_overuse,
                        via_tail_threshold,
                    )
                    stagnant = 0
                else:
                    # Keep accumulated PathFinder history, but branch each
                    # speculative recovery wave from the best routing found
                    # so far. Otherwise repeated bounded rip-ups can walk
                    # steadily away from a good state even though the scalar
                    # best is remembered.
                    current_route_score = self._negotiated_route_score(
                        failed,
                        over_sum,
                        path_node_overuse,
                        conflict_count,
                    )
                    if (
                        best_route_state is not None
                        and best_route_score is not None
                        and best_route_score < current_route_score
                    ):
                        self._restore_routing_state(best_route_state)
                        logger.warning(
                            "[STAGNATION] Rolled back to best iteration %d "
                            "before recovery wave (failed=%d negotiated=%d "
                            "physical=%d)",
                            best_route_iteration,
                            best_route_score[0],
                            best_route_score[1],
                            best_route_score[2],
                        )
                    self.stagnation_counter += 1
                    victims = self._rip_top_k_offenders(k=20)
                    self._last_ripped = victims
                    self._last_stagnation_victims = tuple(
                        sorted(victims)
                    )
                    # Freeze pres_fac for two iterations to let the smaller
                    # hotset settle.
                    self._freeze_pres_fac_until = it + 2
                    logger.warning(
                        "[STAGNATION %d] Ripped %d nets, holding pres_fac "
                        "for 2 iters, ROI margin now +%.1fmm",
                        self.stagnation_counter,
                        len(victims),
                        self.stagnation_counter * 0.6,
                    )
                    stagnant = 0
                    continue

            # STEP 7: Escalate with anti-thrash damper
            if it <= getattr(self, "_freeze_pres_fac_until", 0):
                logger.debug(f"[ITER {it}] Holding pres_fac={pres_fac:.2f} post-rip")
            else:
                # DISABLED: Anti-thrash backoff creates oscillation - use simple exponential growth
                # if over_sum >= 0.995 * prev_over_sum:
                #     stagnant += 1
                # else:
                #     stagnant = 0
                # if stagnant >= 2:
                #     pres_fac = max(1.0, pres_fac * 0.90)
                #     stagnant = 0
                #     logger.info(f"[ANTI-THRASH] stagnant=2 → pres_fac reduced to {pres_fac:.2f}")
                # else:
                #     pres_fac = min(pres_fac * pres_fac_mult, pres_fac_max)

                # Simple exponential growth - no backoff
                pres_fac = min(
                    pres_fac
                    * (pres_fac_mult ** pressure_work_scale),
                    pres_fac_max,
                )
                logger.debug(
                    "[ESCALATE] work_scale=%.2f pres_fac=%.2f",
                    pressure_work_scale,
                    pres_fac,
                )

            # CRITICAL: Update prev_over_sum for next iteration's anti-thrash check
            prev_over_sum = over_sum

        # Negotiated routing can finish on an upswing. Restore the best state
        # before refinement/export so a long run never discards its own best
        # board merely because max_iterations landed at the wrong phase.
        current_route_score = self._negotiated_route_score(
            failed,
            over_sum,
            self._compute_path_node_overuse()[0],
            getattr(self, "_last_barrel_conflict_count", 0),
        )
        restored_best_state = bool(
            best_route_state is not None
            and best_route_score is not None
            and best_route_score < current_route_score
        )
        if restored_best_state:
            self._restore_routing_state(best_route_state)
            failed = sum(
                1 for net_id in tasks
                if not self.net_paths.get(net_id)
            )
            over_sum, over_cnt = self.accounting.compute_overuse(
                router_instance=self
            )
            _, restored_barrel_conflicts = self._detect_barrel_conflicts()
            self._last_barrel_conflict_count = restored_barrel_conflicts
            (
                path_node_overuse,
                path_node_overuse_count,
            ) = self._compute_path_node_overuse()
            self._last_path_node_overuse_total = path_node_overuse
            self._last_path_node_overuse_count = (
                path_node_overuse_count
            )
            negotiated_overuse = over_sum + path_node_overuse
            logger.warning(
                "[BEST-STATE] Restored iteration %d at max_iterations: "
                "failed=%d edge/via=%d nodes=%d negotiated=%d "
                "resources=%d physical=%d",
                best_route_iteration,
                failed,
                over_sum,
                path_node_overuse,
                negotiated_overuse,
                over_cnt,
                restored_barrel_conflicts,
            )

        # If we exited with low overuse (<100), run detail pass
        if 0 < negotiated_overuse <= 100:
            logger.info(
                "[DETAIL PASS] Negotiated overuse=%d at max_iters, "
                "running detail refinement",
                negotiated_overuse,
            )
            detail_result = self._detail_pass(
                tasks,
                negotiated_overuse,
                over_cnt,
            )
            if detail_result['success']:
                detail_result["best_iteration"] = best_route_iteration
                detail_result["restored_best_state"] = restored_best_state
                return detail_result

        # SOFT-FAIL: Analyze if more layers needed
        layer_recommendation = self._analyze_layer_requirements(
            failed,
            over_cnt + path_node_overuse_count,
            negotiated_overuse,
        )

        # Log GPU vs CPU pathfinding statistics
        total_paths = self._gpu_path_count + self._cpu_path_count
        if total_paths > 0:
            gpu_pct = (self._gpu_path_count / total_paths) * 100
            logger.info(f"[GPU-STATS] GPU: {self._gpu_path_count} paths ({gpu_pct:.1f}%), CPU: {self._cpu_path_count} paths ({100-gpu_pct:.1f}%)")

        # Only show warning if routing is actually incomplete
        if failed > 0 or negotiated_overuse > 0:
            logger.warning("="*80)
            logger.warning(f"ROUTING INCOMPLETE: {failed}/{len(tasks)} nets failed ({failed/len(tasks)*100:.1f}%)")
            logger.warning(
                "  Negotiated overuse: %d resources with %d excess "
                "uses (edge/via=%d, node=%d)",
                over_cnt + path_node_overuse_count,
                negotiated_overuse,
                over_sum,
                path_node_overuse,
            )
            if layer_recommendation['needs_more']:
                logger.warning(f"  RECOMMENDATION: Add {layer_recommendation['additional']} more layers (→{layer_recommendation['recommended_total']} total)")
                logger.warning(f"  Reason: {layer_recommendation['reason']}")
            else:
                logger.warning(f"  Current layer count ({self.lattice.layers}) appears adequate")
                logger.warning(f"  Convergence may improve with tuning or may have reached practical limit")
            logger.warning("="*80)
        else:
            logger.info("="*80)
            logger.info(f"ROUTING COMPLETE: All {len(tasks)} nets routed successfully with zero overuse!")
            logger.info("="*80)

        # Final success requires a fresh physical barrel audit as well as
        # edge capacity. A via collision is an electrical short.
        self._rebuild_node_owner()
        _, final_barrel_conflicts = self._detect_barrel_conflicts()
        self._last_barrel_conflict_count = final_barrel_conflicts
        excluded_nets = getattr(self, '_excluded_nets', set())
        success = (
            failed == 0
            and over_sum == 0
            and final_barrel_conflicts == 0
            and not excluded_nets
        )

        if excluded_nets:
            logger.warning(f"[FINAL] {len(excluded_nets)} nets excluded as unroutable: {list(excluded_nets)[:10]}...")

        return {
            'success': success,
            'converged': success,  # Edge convergence = success
            'barrel_conflicts': final_barrel_conflicts,
            'excluded_nets': len(excluded_nets),
            'excluded_net_ids': list(excluded_nets),
            'error_code': None if success else 'ROUTING-FAILED',
            'message': 'Complete' if success else f'{failed} unrouted, {over_cnt} overused',
            'overuse_sum': over_sum,
            'overuse_edges': over_cnt,
            'path_node_overuse_sum': path_node_overuse,
            'path_node_overuse_nodes': path_node_overuse_count,
            'negotiated_overuse_sum': negotiated_overuse,
            'failed_nets': failed,
            'best_iteration': best_route_iteration,
            'restored_best_state': restored_best_state,
            'layer_recommendation': layer_recommendation
        }

    def _analyze_layer_requirements(self, failed_nets: int, overuse_edges: int, overuse_sum: int) -> Dict:
        """Analyze if board needs more layers based on routing failures"""
        current_layers = self.lattice.layers
        total_nets = len(self.net_pad_ids)

        # Calculate failure rate and congestion density
        fail_rate = failed_nets / max(1, total_nets)
        congestion_per_edge = overuse_sum / max(1, overuse_edges) if overuse_edges > 0 else 0

        # Heuristics for layer requirement
        if fail_rate > 0.4 and overuse_edges > 200:
            # High failure rate with significant congestion
            # Estimate: 1 layer per 50 failed nets
            additional = max(4, int(failed_nets / 50))
            return {
                'needs_more': True,
                'additional': additional,
                'recommended_total': current_layers + additional,
                'reason': f'High failure rate ({fail_rate*100:.1f}%) with {overuse_edges} congested edges suggests insufficient layer capacity'
            }
        elif overuse_sum > 800 and overuse_edges > 400:
            # Severe congestion even with partial routing
            return {
                'needs_more': True,
                'additional': 6,
                'recommended_total': current_layers + 6,
                'reason': f'Severe congestion ({overuse_sum} conflicts across {overuse_edges} edges) indicates board density exceeds layer capacity'
            }
        elif fail_rate > 0.3:
            # Moderate failure, try 2-4 more layers
            additional = 4 if fail_rate > 0.35 else 2
            return {
                'needs_more': True,
                'additional': additional,
                'recommended_total': current_layers + additional,
                'reason': f'Moderate routing failure ({fail_rate*100:.1f}%) suggests {additional} additional layers may help'
            }
        else:
            return {
                'needs_more': False,
                'additional': 0,
                'recommended_total': current_layers,
                'reason': f'Failure rate ({fail_rate*100:.1f}%) and overuse ({overuse_edges} edges) within acceptable range for current layer count'
            }

    def _detail_conflict_nets(
        self,
        edge_conflict_nets: Iterable[str] = (),
    ) -> Set[str]:
        """Collect offenders from every resource system for detail routing."""
        return (
            set(edge_conflict_nets)
            | set(getattr(self, "_path_node_conflict_scores", {}))
            | set(self._find_via_pool_offenders())
            | set(getattr(self, "_barrel_conflict_nets", ()))
        )

    def _detail_pass(self, tasks: Dict[str, Tuple[int, int]], initial_overuse: int, initial_edges: int) -> Dict:
        """
        Detail pass: extract conflict subgraph and route only affected nets
        with fine ROI, lower via cost, higher history gain, and max 60 hotset.
        """
        logger.info("[DETAIL] Extracting conflict subgraph...")
        original_state = self._capture_routing_state()

        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)

        # Find overused edges and their neighborhoods (radius ~5-10 edges)
        conflict_edges = set(int(ei) for ei in range(len(over)) if over[ei] > 0)

        # Collect nets that use any conflict edge
        conflict_nets = set()
        for net_id, path in self.net_paths.items():
            graph_path = self._path_without_dynamic_escape_chains(
                net_id, path
            )
            if graph_path and any(
                ei in conflict_edges
                for ei in self._path_to_edges(graph_path)
            ):
                conflict_nets.add(net_id)

        conflict_nets = self._detail_conflict_nets(conflict_nets)
        logger.info(
            "[DETAIL] Found %d nets across edge, via, node, and physical "
            "conflict resources",
            len(conflict_nets),
        )

        if not conflict_nets:
            return {'success': False, 'error_code': 'NO-CONFLICT-NETS'}

        # Build subset of tasks for conflict nets
        conflict_tasks = {nid: tasks[nid] for nid in conflict_nets if nid in tasks}

        # Detail loop: max 10 iterations with aggressive settings
        cfg = self.config
        detail_pres_fac_max = max(
            float(cfg.pres_fac_max),
            float(getattr(
                self,
                "_pres_fac_max_now",
                cfg.pres_fac_max,
            )),
        )
        pres_fac = detail_pres_fac_max * 0.5  # Start high
        best_overuse = initial_overuse

        for detail_it in range(1, 11):
            logger.info(f"[DETAIL {detail_it}/10] pres_fac={pres_fac:.1f}")

            self.accounting.refresh_from_canonical()

            # Update costs with lower via cost and higher history gain
            via_cost_mult = 0.3  # Much lower via cost for detail pass
            self.accounting.update_costs(
                self.graph.base_costs, pres_fac, cfg.hist_cost_weight * 1.5,
                via_cost_multiplier=via_cost_mult,
                base_cost_weight=cfg.base_cost_weight
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
            over_sum, over_cnt = self.accounting.compute_overuse(router_instance=self)
            self._rebuild_node_owner()
            _, barrel_conflicts = self._detect_barrel_conflicts()
            self._last_barrel_conflict_count = barrel_conflicts
            (
                path_node_overuse,
                _path_node_overuse_count,
            ) = self._compute_path_node_overuse()
            negotiated_overuse = over_sum + path_node_overuse
            conflict_nets = self._detail_conflict_nets(conflict_nets)
            conflict_tasks.update({
                net_id: tasks[net_id]
                for net_id in conflict_nets
                if net_id in tasks
            })

            logger.info(
                "[DETAIL %d/10] edge/via=%d nodes=%d negotiated=%d "
                "resources=%d physical=%d",
                detail_it,
                over_sum,
                path_node_overuse,
                negotiated_overuse,
                over_cnt,
                barrel_conflicts,
            )

            if negotiated_overuse == 0:
                unrouted = {
                    net_id
                    for net_id in tasks
                    if not self.net_paths.get(net_id)
                }
                if unrouted:
                    conflict_nets = unrouted
                    conflict_tasks.update({
                        net_id: tasks[net_id]
                        for net_id in unrouted
                    })
                    logger.info(
                        "[DETAIL] Edge-clean state still has %d "
                        "unrouted nets; continuing",
                        len(unrouted),
                    )
                    continue
                if barrel_conflicts:
                    logger.info(
                        "[DETAIL] Graph-clean state has %d physical "
                        "conflicts; continuing",
                        barrel_conflicts,
                    )
                    continue
                logger.info("[DETAIL] SUCCESS: Zero overuse achieved")
                # Log GPU vs CPU pathfinding statistics
                total_paths = self._gpu_path_count + self._cpu_path_count
                if total_paths > 0:
                    gpu_pct = (self._gpu_path_count / total_paths) * 100
                    logger.info(f"[GPU-STATS] GPU: {self._gpu_path_count} paths ({gpu_pct:.1f}%), CPU: {self._cpu_path_count} paths ({100-gpu_pct:.1f}%)")
                return {'success': True, 'paths': self.net_paths}

            if negotiated_overuse < best_overuse:
                best_overuse = negotiated_overuse
            else:
                # No improvement: escalate and continue
                pass

            # Update history for conflict edges only
            self.accounting.update_history(
                cfg.hist_gain * 2.0,  # Double history gain in detail pass
                base_costs=self.graph.base_costs,
                history_cap_multiplier=15.0,
                decay_factor=0.98  # Use decay in detail pass to allow redistribution
            )

            pres_fac = min(
                pres_fac * 1.5,
                detail_pres_fac_max,
            )

        # Detail pass exhausted
        logger.warning(f"[DETAIL] Failed to reach zero: final overuse={best_overuse}")
        # A refinement attempt is speculative. Do not return a lower-overuse
        # state that achieved it by dropping nets.
        self._restore_routing_state(original_state)
        return {'success': False, 'error_code': 'DETAIL-INCOMPLETE', 'overuse_sum': best_overuse}

    def _capture_routing_state(self) -> Dict:
        """Copy the export-relevant negotiated state compactly in memory."""
        return {
            "paths": {
                net_id: list(path)
                for net_id, path in self.net_paths.items()
            },
            # Portal objects are mutable during retargeting. A shallow mapping
            # copy would silently mutate the remembered best state.
            "selected_portals": copy.deepcopy(
                self.net_selected_portals
            ),
            "portal_layers": dict(self.net_portal_layers),
        }

    def _restore_routing_state(self, state: Dict) -> None:
        """Restore paths and rebuild every derived resource index."""
        self.net_paths.clear()
        self.net_paths.update({
            net_id: list(path)
            for net_id, path in state["paths"].items()
        })
        self.net_selected_portals.clear()
        self.net_selected_portals.update(copy.deepcopy(
            state["selected_portals"]
        ))
        self.net_portal_layers.clear()
        self.net_portal_layers.update(state["portal_layers"])
        self.accounting.canonical.clear()
        self.accounting.present.fill(0)
        self._net_to_edges.clear()
        self._edge_to_nets.clear()
        for net_id, path in self.net_paths.items():
            if not path or len(path) < 2:
                continue
            graph_path = self._path_without_dynamic_escape_chains(
                net_id, path
            )
            edge_indices = self._path_to_edges(graph_path)
            for edge_idx in edge_indices:
                self.accounting.canonical[edge_idx] = (
                    self.accounting.canonical.get(edge_idx, 0) + 1
                )
            self._update_net_edge_tracking(net_id, edge_indices)
        self.accounting.refresh_from_canonical()
        self._rebuild_via_usage_from_committed()
        self._rebuild_node_owner()
        self._rebuild_path_node_use()
        self._rebuild_escape_occupancy()

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

        # When repairing barrel conflicts, commit the barrel owner before the
        # track that crosses it. Reversing that order recreates an unavoidable
        # portal via after the victim has already routed through its cell.
        owner_nets = getattr(self, "_barrel_owner_nets", set())
        victim_nets = getattr(self, "_barrel_victim_nets", set())
        scores.sort(
            key=lambda item: (
                2 if item[1] in owner_nets and item[1] not in victim_nets
                else 1 if item[1] in owner_nets
                else 0,
                item[0],
                item[1],
            ),
            reverse=True,
        )

        # Never rotate across barrel-role boundaries: moving even one forced
        # barrel owner behind its crossing track recreates the same short.
        # Ordinary congestion-only passes retain the tie-breaking rotation.
        rotation = 0
        if not owner_nets and not victim_nets:
            rng = random.Random(
                42 + int(getattr(self, "iteration", 0))
            )
            rotation = rng.randint(
                0, min(5, len(scores) // 10)
            )
        ordered = [nid for _, nid in scores]
        if rotation > 0:
            ordered = ordered[rotation:] + ordered[:rotation]

        return ordered

    def _route_all(self, tasks: Dict[str, Tuple[int, int]], all_tasks: Dict[str, Tuple[int, int]] = None, pres_fac: float = 1.0, iteration: int = 0) -> Tuple[int, int]:
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
        if (
            cfg.portal_enabled
            and not hasattr(self, "_escape_preferred_portals")
        ):
            self._plan_escape_assignment()
        self._rebuild_escape_occupancy()

        # Start with costs from iteration-level update
        # NOTE: For GPU, keep as CuPy array (don't .get() to CPU)
        # Per-net updates will refresh this reference
        costs = self.accounting.total_cost  # Keep on GPU if use_gpu=True

        # History and base costs stay fixed for the iteration. Present costs
        # track live occupancy so two nets do not select the same newly freed
        # edge from an identical stale congestion view.
        live_present_costs = bool(
            getattr(cfg, "live_present_costs", True)
        )
        if live_present_costs:
            self.accounting.begin_live_present_costs()
        
        # CRITICAL: Verify sequential mode (SEQUENTIAL_ALL env var check)
        force_sequential = os.getenv("SEQUENTIAL_ALL") == "1"
        if force_sequential:
            logger.info(f"[SEQUENTIAL_ALL] Sequential mode ENABLED via environment variable")
        logger.info(f"[ROUTING MODE] Using sequential routing for {total} nets (PathFinder algorithm)")

        for idx, net_id in enumerate(ordered_nets):
            net_start_time = time.time()
            src, dst = tasks[net_id]

            # Show progress more frequently in iteration 1 (greedy routing)
            if iteration == 1:
                # Show every 25 nets in iteration 1
                if (idx + 1) % 25 == 0 or idx == 0:
                    logger.warning(f"[ITER 1 - GREEDY] Routing {idx+1}/{total} nets...")
            elif idx % 50 == 0 and total > 50:
                logger.debug(f"  Routing net {idx+1}/{total}")

            # Only clear if we're actually re-routing this net
            if net_id in self.net_paths and self.net_paths[net_id]:
                self._clear_escape_occupancy(net_id)
                # Use cached edges if available, otherwise compute
                if net_id in self._net_to_edges:
                    old_edges = self._net_to_edges[net_id]
                else:
                    old_edges = self._path_to_edges(self.net_paths[net_id])
                self._clear_via_barrel_ownership_for_path(
                    net_id, self.net_paths[net_id]
                )
                self._clear_path_node_use(self.net_paths[net_id])
                self.accounting.clear_path(old_edges)
                if live_present_costs:
                    self.accounting.refresh_live_present_costs(old_edges)
                # Clear old tracking before re-routing
                self._clear_net_edge_tracking(net_id)

            # Calculate adaptive ROI radius based on net length
            src_x, src_y, src_z = self.lattice.idx_to_coord(src)
            dst_x, dst_y, dst_z = self.lattice.idx_to_coord(dst)
            manhattan_dist = abs(dst_x - src_x) + abs(dst_y - src_y)

            # Adaptive radius: 120% of Manhattan distance + minimum 10-step buffer
            # Increased max from 150 to 800 to handle full-board routes
            adaptive_radius = max(40, min(int(manhattan_dist * 1.2) + 10, 800))
            if manhattan_dist > 800:
                logger.warning(f"Net {net_id}: manhattan_dist={manhattan_dist} exceeds max radius=800!")

            # Check if we have portals for this net
            use_portals = cfg.portal_enabled and net_id in self.net_pad_ids
            src_seeds = []
            dst_targets = []
            src_seed_portals = {}
            dst_target_portals = {}

            if use_portals:
                src_pad_id, dst_pad_id = self.net_pad_ids[net_id]
                src_portal = self.portals.get(src_pad_id)
                dst_portal = self.portals.get(dst_pad_id)

                if src_portal and dst_portal:
                    # Negotiate both escape position and entry layer.
                    src_seeds, src_seed_portals = (
                        self._get_pad_portal_seeds(
                            src_pad_id, current_net=net_id
                        )
                    )
                    dst_targets, dst_target_portals = (
                        self._get_pad_portal_seeds(
                            dst_pad_id, current_net=net_id
                        )
                    )
                else:
                    use_portals = False


            # ═══════════════════════════════════════════════════════════════
            # GPU SUPERSOURCE FAST PATH (ATTEMPT BEFORE ROI EXTRACTION)
            # ═══════════════════════════════════════════════════════════════
            # Try full-graph GPU pathfinding with supersource/supersink seeds
            # This skips ROI extraction entirely and routes on full graph
            # If it fails or GPU not available, fall back to standard ROI routing
            
            gpu_fast_path_used = False
            gpu_fullgraph_failed = False
            # Full-graph routing with owner-aware bitmap filtering
            if use_portals and hasattr(self.solver, 'gpu_solver') and self.solver.gpu_solver:
                try:
                    import numpy as np
                    import cupy as cp

                    # Check if costs are on GPU
                    costs_on_gpu = hasattr(costs, 'device')

                    if costs_on_gpu:
                        logger.info(f"[GPU-SEEDS] Attempting GPU supersource routing for net {net_id}")

                        # Convert portal seeds to plain node arrays
                        src_node_array = self._build_routing_seeds(src_seeds)
                        dst_node_array = self._build_routing_seeds(dst_targets)
                        src_seed_costs = np.asarray(
                            [cost for _, cost in src_seeds],
                            dtype=np.float32,
                        )
                        dst_target_costs = np.asarray(
                            [cost for _, cost in dst_targets],
                            dtype=np.float32,
                        )

                        # Ownership is a negotiated node cost, not a hard
                        # wall. Keep that cost on every candidate terminal:
                        # clearing it for all source/destination alternatives
                        # makes foreign via barrels free to enter precisely
                        # where dense connector fields need the strongest
                        # discrimination.
                        owner_penalty = self._build_owner_penalty_gpu(
                            net_id
                        )
                        route_costs = costs
                        if getattr(
                            self, "_freeze_selected_portals", False
                        ):
                            forbidden_edges = (
                                self._portal_cleanup_foreign_edges(
                                    net_id
                                )
                            )
                            if forbidden_edges.size:
                                route_costs = costs.copy()
                                route_costs[cp.asarray(
                                    forbidden_edges,
                                    dtype=cp.int32,
                                )] += np.float32(getattr(
                                    cfg,
                                    "portal_cleanup_edge_penalty",
                                    1_000_000.0,
                                ))

                        if len(src_node_array) > 0 and len(dst_node_array) > 0:
                            # Call GPU supersource pathfinding with owner-aware cost.
                            gpu_start = time.time()
                            path = self.solver.gpu_solver.find_path_fullgraph_gpu_seeds(
                                costs=route_costs,
                                src_seeds=src_node_array,
                                dst_targets=dst_node_array,
                                ub_hint=None,
                                src_seed_costs=src_seed_costs,
                                dst_target_costs=dst_target_costs,
                                node_penalty=owner_penalty,
                                allowed_bitmap=None,
                                use_bitmap=False,
                            )
                            gpu_time = time.time() - gpu_start

                            if path and len(path) > 1:
                                logger.info(f"[GPU-SEEDS] SUCCESS! Path found in {gpu_time:.3f}s ({len(path)} nodes)")
                                gpu_fast_path_used = True
                                
                                # Determine entry and exit layers from path
                                if self.solver.plane_size:
                                    entry_layer = path[0] // self.solver.plane_size
                                    exit_layer = path[-1] // self.solver.plane_size
                                else:
                                    entry_layer = exit_layer = 0

                                selected_src_portal = (
                                    src_seed_portals.get(
                                        path[0], src_portal
                                    )
                                )
                                selected_dst_portal = (
                                    dst_target_portals.get(
                                        path[-1], dst_portal
                                    )
                                )
                                path = self._attach_portal_vias(
                                    path,
                                    selected_src_portal,
                                    selected_dst_portal,
                                )

                                self.net_selected_portals[net_id] = (
                                    selected_src_portal,
                                    selected_dst_portal,
                                )
                                if (
                                    entry_layer is not None
                                    and exit_layer is not None
                                ):
                                    self.net_portal_layers[net_id] = (
                                        entry_layer, exit_layer
                                    )
                                graph_path = (
                                    self._path_without_dynamic_escape_chains(
                                        net_id, path
                                    )
                                )

                                # Commit path and continue to next net
                                edge_indices = self._path_to_edges(graph_path)
                                self.accounting.commit_path(edge_indices)
                                if live_present_costs:
                                    self.accounting.refresh_live_present_costs(
                                        edge_indices
                                    )

                                self.net_paths[net_id] = path

                                # CRITICAL: Mark via barrel ownership IMMEDIATELY for next net in same iteration!
                                self._mark_via_barrel_ownership_for_path(net_id, path)
                                self._mark_path_node_use(path)

                                self._mark_escape_occupancy(
                                    net_id,
                                    self.net_selected_portals[net_id],
                                )
                                self._update_net_edge_tracking(net_id, edge_indices)
                                routed_this_pass += 1
                                continue  # Skip ROI extraction and CPU routing
                            else:
                                gpu_fullgraph_failed = True
                                logger.info(
                                    "[GPU-SEEDS] No full-graph path found; "
                                    "falling back to cost-based ROI routing"
                                )
                        else:
                            logger.warning(f"[GPU-SEEDS] Empty seed arrays, skipping GPU fast path")
                except Exception as e:
                    gpu_fullgraph_failed = True
                    logger.warning(
                        f"[GPU-SEEDS] GPU fast path failed: {e}; "
                        "falling back to cost-based ROI routing"
                    )

            # For a long net on a huge board, the "ROI" below is the entire
            # graph.  Falling through to CPU multisource Dijkstra can take
            # many minutes per miss.  Record a normal negotiated failure and
            # retry after costs/portals change.  Small graphs retain the CPU
            # correctness fallback.
            if (
                gpu_fullgraph_failed
                and self.N >= int(getattr(
                    cfg, "gpu_fullgraph_fail_fast_nodes", 1_000_000
                ))
            ):
                logger.warning(
                    f"[GPU-SEEDS] Net {net_id}: skipping CPU full-graph "
                    f"fallback for {self.N:,} nodes"
                )
                failed_this_pass += 1
                self.net_paths[net_id] = []
                self.net_selected_portals.pop(net_id, None)
                self._clear_net_edge_tracking(net_id)
                if cfg.portal_enabled and net_id in self.net_pad_ids:
                    self.net_portal_failures[net_id] += 1
                    if (
                        self.net_portal_failures[net_id]
                        >= cfg.portal_retarget_patience
                    ):
                        self._retarget_portals_for_net(net_id)
                        self.net_portal_failures[net_id] = 0
                continue
            
            # If GPU fast path succeeded, we already continued to next net above
            # Otherwise, proceed with standard ROI routing below
            # ═══════════════════════════════════════════════════════════════
            # HYBRID ROI/FULL-GRAPH ROUTING (BACKPLANE-OPTIMIZED)
            # ═══════════════════════════════════════════════════════════════
            # Strategy:
            # - SHORT NETS (<125 steps = 50mm): Use ROI → 10× faster, <50 iterations
            # - LONG NETS (≥125 steps): Use full graph → necessary for board-spanning
            # Rationale: Backplanes have power/ground/clock spanning entire board
            roi_start = time.time()

            # Distance threshold: 125 steps @ 0.4mm = 50mm
            ROI_THRESHOLD_STEPS = 125
            use_roi_extraction = manhattan_dist < ROI_THRESHOLD_STEPS

            import numpy as np
            if use_roi_extraction:
                # SHORT NET: Use focused ROI for fast routing
                # Determine src/dst for ROI center
                if use_portals and src_portal and dst_portal:
                    roi_src_node = self.lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.entry_layer)
                    roi_dst_node = self.lattice.node_idx(dst_portal.x_idx, dst_portal.y_idx, dst_portal.entry_layer)
                else:
                    roi_src_node = src
                    roi_dst_node = dst

                # Adaptive radius: 150% of distance for detours
                adaptive_radius = max(40, int(manhattan_dist * 1.5) + 10)

                # Gather portal seeds if available
                portal_seeds = None
                if use_portals and src_portal and dst_portal:
                    portal_seeds = (
                        (src_seeds or []) + (dst_targets or [])
                    )

                # Extract ROI
                roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                    roi_src_node, roi_dst_node,
                    initial_radius=adaptive_radius,
                    stagnation_bonus=roi_margin_bonus,
                    portal_seeds=portal_seeds
                )

                if idx % 100 == 0:
                    logger.info(f"[HYBRID] Net {net_id}: SHORT ({manhattan_dist} steps) → ROI ({len(roi_nodes):,} nodes)")
            else:
                # LONG NET: Use full graph (board-spanning)
                # Cached identity arrays: rebuilding these per long net cost
                # two 32MB allocations each on an 8M-node monster graph.
                if getattr(self, '_full_roi_cache', None) is None or \
                        len(self._full_roi_cache) != self.N:
                    self._full_roi_cache = np.arange(self.N, dtype=np.int32)
                roi_nodes = self._full_roi_cache
                global_to_roi = self._full_roi_cache

                if idx % 100 == 0:
                    logger.info(f"[HYBRID] Net {net_id}: LONG ({manhattan_dist} steps) → FULL GRAPH ({self.N:,} nodes)")

            roi_time = time.time() - roi_start

            # Ensure portal seeds are in ROI (important for multi-layer routing)
            portal_setup_start = time.time()
            nodes_to_add = []

            # Add portal seed nodes if they're not already in ROI
            if use_portals and src_seeds and dst_targets:
                for global_node, _ in src_seeds:
                    if global_to_roi[global_node] < 0:
                        nodes_to_add.append(global_node)
                for global_node, _ in dst_targets:
                    if global_to_roi[global_node] < 0:
                        nodes_to_add.append(global_node)

            # Add missing portal nodes to ROI
            if nodes_to_add:
                roi_nodes = np.append(roi_nodes, nodes_to_add)
                # Rebuild mapping to include new nodes (node-indexed: size N)
                global_to_roi = np.full(self.N, -1, dtype=np.int32)
                global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)

            # OWNERSHIP-AS-COST: price nodes owned by OTHER nets instead of
            # removing them from the ROI. The old hard filter stripped nets'
            # own endpoints when a neighbor's barrel claimed them ("BUG: src
            # not in ROI") and forced ever-wilder detours - barrel conflicts
            # are congestion, and congestion is negotiated, not walled off.
            owner_penalty = self._build_owner_penalty(roi_nodes, net_id)

            # Final sanity check
            if global_to_roi[src] < 0:
                logger.error(f"BUG: src {src} not in ROI after all additions!")
            if global_to_roi[dst] < 0:
                logger.error(f"BUG: dst {dst} not in ROI after all additions!")

            portal_setup_time = time.time() - portal_setup_start

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

            pathfinding_start = time.time()
            if use_portals:
                # Route with multi-source/multi-sink using portal seeds
                result = self.solver.find_path_multisource_multisink(
                    src_seeds, dst_targets, costs, roi_nodes, global_to_roi,
                    node_penalty=owner_penalty
                )
                if result:
                    path, entry_layer, exit_layer = result

            # Fallback to normal routing if portals not available or failed
            if not use_portals or not path:
                if idx == 0 and use_portals:
                    logger.info(f"[DEBUG] Portal routing failed, falling back to normal routing")
                path = self.solver.find_path_roi(src, dst, costs, roi_nodes, global_to_roi,
                                                 node_penalty=owner_penalty)
            pathfinding_time = time.time() - pathfinding_start

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
                        # CRITICAL FIX: Use entry_layer (routing layer) not pad_layer (F.Cu)!
                        src_portal_node = self.lattice.node_idx(src_portal.x_idx, src_portal.y_idx, src_portal.entry_layer)
                        dst_portal_node = self.lattice.node_idx(dst_portal.x_idx, dst_portal.y_idx, dst_portal.entry_layer)
                        roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                            src_portal_node, dst_portal_node, initial_radius=fallback_radius, stagnation_bonus=roi_margin_bonus * 2
                        )
                        owner_penalty = self._build_owner_penalty(roi_nodes, net_id)
                        result = self.solver.find_path_multisource_multisink(
                            src_seeds, dst_targets, costs, roi_nodes, global_to_roi,
                            node_penalty=owner_penalty
                        )
                        if result:
                            path, entry_layer, exit_layer = result
                else:
                    # Use pad nodes for larger ROI
                    roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
                        src, dst, initial_radius=fallback_radius, stagnation_bonus=roi_margin_bonus * 2
                    )
                    owner_penalty = self._build_owner_penalty(roi_nodes, net_id)
                    path = self.solver.find_path_roi(src, dst, costs, roi_nodes, global_to_roi,
                                                     node_penalty=owner_penalty)

                self.full_graph_fallback_count += 1

            if path and len(path) > 1:
                if use_portals and src_portal and dst_portal:
                    selected_src_portal = src_seed_portals.get(
                        path[0], src_portal
                    )
                    selected_dst_portal = dst_target_portals.get(
                        path[-1], dst_portal
                    )
                    path = self._attach_portal_vias(
                        path,
                        selected_src_portal,
                        selected_dst_portal,
                    )
                    self.net_selected_portals[net_id] = (
                        selected_src_portal,
                        selected_dst_portal,
                    )
                    if (
                        entry_layer is not None
                        and exit_layer is not None
                    ):
                        self.net_portal_layers[net_id] = (
                            entry_layer, exit_layer
                        )
                    self._mark_escape_occupancy(
                        net_id,
                        self.net_selected_portals[net_id],
                    )
                graph_path = (
                    self._path_without_dynamic_escape_chains(net_id, path)
                    if use_portals else path
                )
                edge_indices = self._path_to_edges(graph_path)
                self.accounting.commit_path(edge_indices)  # bumps present for next iteration
                if live_present_costs:
                    self.accounting.refresh_live_present_costs(
                        edge_indices
                    )

                self.net_paths[net_id] = path

                # CRITICAL: Mark via barrel ownership IMMEDIATELY for next net in same iteration!
                self._mark_via_barrel_ownership_for_path(net_id, path)
                self._mark_path_node_use(path)

                # Store portal entry/exit layers if using portals
                if use_portals and entry_layer is not None and exit_layer is not None:
                    self.net_portal_layers[net_id] = (entry_layer, exit_layer)
                # Update edge-to-nets tracking
                self._update_net_edge_tracking(net_id, edge_indices)
                routed_this_pass += 1
            else:
                failed_this_pass += 1
                self.net_paths[net_id] = []
                self.net_selected_portals.pop(net_id, None)
                # Clear tracking for failed nets
                self._clear_net_edge_tracking(net_id)

                # Track portal failures and retarget if needed
                if cfg.portal_enabled and net_id in self.net_pad_ids:
                    self.net_portal_failures[net_id] += 1
                    if self.net_portal_failures[net_id] >= cfg.portal_retarget_patience:
                        # Retarget portals for this net
                        self._retarget_portals_for_net(net_id)
                        self.net_portal_failures[net_id] = 0  # Reset counter

            # Log timing for first net and periodically thereafter
            net_total_time = time.time() - net_start_time
            if idx == 0 or (idx < 10) or (idx % 50 == 0):
                logger.info(f"[TIMING] Net {idx+1}/{total} ({net_id}): ROI={roi_time:.3f}s (size={len(roi_nodes)}), Portal={portal_setup_time:.3f}s, Path={pathfinding_time:.3f}s, Total={net_total_time:.3f}s")

        # Count total routed/failed across all nets
        total_routed = sum(1 for path in self.net_paths.values() if path)
        total_failed = len(all_tasks) - total_routed

        return total_routed, total_failed

    def _compute_layer_bias(self, accountant, graph, num_layers: int, alpha: float = 0.9, max_boost: float = 1.8):
        """
        Compute per-layer multiplicative bias (shape [L]) based on overuse distribution.
        Hot layers get bias > 1.0, cool layers stay at 1.0.

        Args:
            accountant: EdgeAccountant with present/capacity arrays
            graph: CSRGraph with edge_layer mapping
            num_layers: Number of layers
            alpha: EMA smoothing factor (0..1, higher = smoother)
            max_boost: Maximum bias multiplier (1.0 to max_boost)

        Returns:
            Layer bias array or None if layer mapping unavailable
        """
        xp = accountant.xp

        # Check if edge_layer mapping exists
        edge_layer = getattr(graph, "edge_layer_gpu", None) if accountant.use_gpu else getattr(graph, "edge_layer", None)
        if edge_layer is None:
            return None

        # Get overuse array - use smoothed present if available
        present_for_bias = getattr(accountant, 'present_ema', accountant.present)
        over = xp.maximum(0, present_for_bias - accountant.capacity)

        # Sum overuse per layer (ONE bincount - very fast)
        per_layer_over = xp.bincount(edge_layer, weights=over, minlength=num_layers)

        # Normalize to create bias factors
        maxv = float(per_layer_over.max().get() if accountant.use_gpu else per_layer_over.max())
        if maxv <= 1e-9:
            raw_bias = xp.ones(num_layers, dtype=xp.float32)
        else:
            shortfall = per_layer_over / maxv
            # Baseline bias from document
            raw_bias = 1.0 + 0.75 * shortfall  # Bias = 1.0 to 1.75

        depth_alpha = max(
            0.0, float(getattr(self.config, "layer_depth_bias", 0.0))
        )
        if depth_alpha:
            raw_bias = (
                raw_bias
                + depth_alpha
                * xp.arange(num_layers, dtype=xp.float32)
            )

        # EMA smoothing to prevent oscillation
        if not hasattr(self, "_layer_bias_ema"):
            self._layer_bias_ema = raw_bias.astype(xp.float32)
        else:
            self._layer_bias_ema = (alpha * self._layer_bias_ema + (1.0 - alpha) * raw_bias).astype(xp.float32)

        # Clamp to prevent extreme penalties
        depth_ceiling = max_boost + depth_alpha * max(
            0, num_layers - 1
        )
        self._layer_bias_ema = xp.clip(
            self._layer_bias_ema, 1.0, depth_ceiling
        )

        return self._layer_bias_ema

    def _via_depart_discount(self, z_from: int, z_to: int) -> float:
        """
        Compute via cost discount for leaving hot layers.
        Encourages routing to move off congested layers toward cooler layers.
        AGGRESSIVE discounting to force layer spreading.

        Args:
            z_from: Source layer
            z_to: Target layer

        Returns:
            Discount multiplier (0.2-1.0), lower = cheaper via
        """
        # Only apply discount if we have layer bias data
        if not hasattr(self, '_layer_bias_ema') or self._layer_bias_ema is None:
            return 1.0

        # Get bias values (on CPU for simplicity, these are small arrays)
        if hasattr(self._layer_bias_ema, 'get'):
            bias_cpu = self._layer_bias_ema.get()
        else:
            bias_cpu = self._layer_bias_ema

        if z_from >= len(bias_cpu) or z_to >= len(bias_cpu):
            return 1.0

        bf = float(bias_cpu[z_from])  # Source layer bias
        bt = float(bias_cpu[z_to])    # Target layer bias

        same_orientation = (
            self.lattice.get_legal_axis(z_from)
            == self.lattice.get_legal_axis(z_to)
        )

        # Calculate discount based on bias difference
        delta = max(0.0, bf - bt)

        if same_orientation and delta > 0.15:
            # Moderate: If leaving hot H layer for cool H layer (or V→V), give good discount
            # This encourages spreading within same routing direction
            discount = 1.0 - min(0.45, 0.8 * delta)  # Up to 45% off
        elif delta > 0.1:
            # Standard: Leaving any hot layer for cooler layer
            discount = 1.0 - min(0.35, 0.6 * delta)  # Up to 35% off
        else:
            # Small benefit
            discount = 1.0 - min(0.15, 0.4 * delta)  # Up to 15% off

        return max(0.55, discount)  # Never go below 0.55 (45% off maximum)

    def _retarget_portals_for_net(self, net_id: str):
        """Retarget both portals when a net fails repeatedly."""
        if net_id not in self.net_pad_ids:
            return
        for pad_id in self.net_pad_ids[net_id]:
            self._retarget_portal(pad_id)

    def _retarget_portal(self, pad_id: str):
        """Move one portal to a new candidate position, safely.

        Strategy 0: flip to the other side of the pad.
        Strategy 1: adjust length toward portal_delta_pref.
        The move is rejected (attempt still consumed) if the new cell is
        off-lattice or already claimed by another portal.

        NOTE: the previous version flipped direction and then applied
        y - 2*direction*delta with the ALREADY-flipped direction, moving the
        via 2*delta FURTHER on the original side - off the lattice for edge
        pads (crashed ROI extraction with out-of-range node indices).
        """
        portal = self.portals.get(pad_id)
        if portal is None:
            return
        if portal.dynamic_entry:
            return

        if portal.retarget_count == 0:
            # Flip to the other side: y_pad - d*delta  ==  y_old + 2*(-d)*delta
            new_dir = -portal.direction
            new_delta = portal.delta_steps
            new_y = portal.y_idx + 2 * new_dir * portal.delta_steps
        elif portal.retarget_count == 1:
            new_dir = portal.direction
            new_delta = self.config.portal_delta_pref
            if new_delta == portal.delta_steps:
                return
            new_y = portal.y_idx + new_dir * (new_delta - portal.delta_steps)
        else:
            return

        portal.retarget_count += 1

        if not (0 <= new_y < self.lattice.y_steps):
            logger.debug(f"Retarget rejected for {pad_id}: y={new_y} off-lattice")
            return

        cells = getattr(getattr(self, 'escape_planner', None),
                        '_occupied_portal_cells', None)
        if cells is not None:
            if (portal.x_idx, new_y) in cells:
                logger.debug(f"Retarget rejected for {pad_id}: cell occupied")
                return
            cells.discard((portal.x_idx, portal.y_idx))
            cells.add((portal.x_idx, new_y))

        portal.direction = new_dir
        portal.delta_steps = new_delta
        portal.y_idx = new_y
        logger.debug(f"Retargeted portal for {pad_id}: dir={new_dir} delta={new_delta} y={new_y}")

    def _rebuild_usage_from_committed_nets(self, keep_net_ids: Set[str]):
        """
        Refresh device usage from the canonical edge counts.

        commit_path/clear_path maintain canonical synchronously with
        net_paths. Rewalking every path here performed one GPU scalar update
        per edge and dominated large-board iterations.
        """
        self.accounting.refresh_from_canonical()
        logger.debug(
            f"[USAGE] Refreshed {len(keep_net_ids)} committed nets "
            "from canonical counts"
        )

    def _update_net_edge_tracking(self, net_id: str, edge_indices: List[int]):
        """Update edge-to-nets tracking when a net is routed"""
        # Clear old tracking for this net
        self._clear_net_edge_tracking(net_id)

        # Store new edges for this net
        self._net_to_edges[net_id] = edge_indices

        # Update reverse mapping
        for ei in edge_indices:
            self._edge_to_nets[ei].add(net_id)

    @staticmethod
    def _history_hotset_cap(total_overuse: float) -> int:
        """Scale reroute waves without using the tail policy globally.

        Small caps protect a nearly-clean route from destructive churn. A
        monster route with tens of thousands of exact node conflicts is a
        different regime: paying the full-graph accounting cost to move only
        100 of 8,192 nets makes tiny minima look like useful convergence.
        Conflict-aware selection below keeps the larger waves one-sided.
        """
        if total_overuse <= 8:
            return 16
        if total_overuse <= 32:
            return 32
        if total_overuse <= 128:
            return 64
        if total_overuse <= 2_048:
            return 100
        if total_overuse <= 16_384:
            return 180
        return 256

    @staticmethod
    def _rolling_progress_insufficient(
        values,
        window: int = 5,
        minimum_fraction: float = 0.025,
        minimum_overuse: int = 16_384,
    ) -> Tuple[bool, Optional[float]]:
        """Return whether the best rolling descent is operationally slow."""
        window = max(1, int(window))
        if len(values) < window + 1:
            return False, None
        start = float(values[-window - 1])
        if start <= max(0, int(minimum_overuse)):
            return False, None
        best_later = min(map(float, values[-window:]))
        improvement_fraction = max(
            0.0,
            (start - best_later) / max(1.0, start),
        )
        threshold = max(0.0, float(minimum_fraction))
        return improvement_fraction < threshold, improvement_fraction

    @staticmethod
    def _pressure_work_scale(
        routed_task_count: int,
        reference_hotset: int = 100,
        maximum_scale: float = 2.0,
    ) -> float:
        """Return bounded equivalent pressure steps for a selective pass."""
        reference = max(1, int(reference_hotset))
        maximum = max(1.0, float(maximum_scale))
        return min(
            maximum,
            max(1.0, max(0, int(routed_task_count)) / reference),
        )

    def _effective_history_hotset_cap(
        self,
        total_overuse: float,
    ) -> int:
        """Apply the temporary rate-based severe-wave expansion."""
        base = self._history_hotset_cap(total_overuse)
        severe_threshold = int(getattr(
            self.config,
            "slow_progress_min_overuse",
            16_384,
        ))
        if (
            total_overuse > severe_threshold
            and self.iteration <= int(getattr(
                self,
                "_hotset_rate_boost_until",
                0,
            ))
        ):
            initial_cap = max(
                1,
                int(getattr(
                    self.config,
                    "slow_progress_hotset_cap",
                    512,
                )),
            )
            maximum_cap = max(
                initial_cap,
                int(getattr(
                    self.config,
                    "slow_progress_hotset_cap_max",
                    initial_cap,
                )),
            )
            growth_after = max(
                1,
                int(getattr(
                    self.config,
                    "slow_progress_hotset_growth_after",
                    2,
                )),
            )
            event_count = max(
                1,
                int(getattr(
                    self,
                    "_slow_progress_event_count",
                    1,
                )),
            )
            growth_steps = max(0, event_count - growth_after + 1)
            boosted_cap = min(
                maximum_cap,
                initial_cap * (2 ** growth_steps),
            )
            return max(
                base,
                boosted_cap,
            )
        return base

    @staticmethod
    def _hotset_exploration_fraction(total_overuse: float) -> float:
        """Spend less of a severe-congestion wave on random search."""
        if total_overuse > 16_384:
            return 0.15
        if total_overuse > 2_048:
            return 0.25
        return 0.40

    @staticmethod
    def _conflict_pair_coverage(
        selected,
        conflict_pairs,
    ) -> Tuple[int, int]:
        """Return distinct live conflict pairs and pairs touched by a wave."""
        selected_set = set(selected)
        unique_pairs = set()
        covered = 0
        for first, second in conflict_pairs or ():
            if first == second:
                continue
            pair = frozenset((first, second))
            if len(pair) != 2 or pair in unique_pairs:
                continue
            unique_pairs.add(pair)
            if first in selected_set or second in selected_set:
                covered += 1
        return len(unique_pairs), covered

    def _record_hotset_conflict_coverage(self, hotset) -> None:
        """Journal how much of the current node-conflict graph is touched."""
        pair_count, covered = self._conflict_pair_coverage(
            hotset,
            getattr(self, "_path_node_conflict_pairs", ()),
        )
        self._last_hotset_conflict_pair_count = pair_count
        self._last_hotset_conflict_pairs_covered = covered
        self._last_hotset_conflict_pair_coverage_fraction = (
            covered / pair_count if pair_count else 0.0
        )

    @staticmethod
    def _select_conflict_aware_hotset(
        ranked_candidates: List[str],
        conflict_pairs,
        cap: int,
        exploration_fraction: float,
        rng,
    ) -> List[str]:
        """Select a conflict-covering wave plus bounded exploration.

        _route_all clears and recommits each selected net sequentially, with
        live edge and node occupancy refreshed between nets.  Selecting a
        global independent set is therefore unnecessary: on a dense conflict
        component it can collapse a nominally large wave to only a handful
        of nets.  Instead, greedily cover the greatest number of still-live
        conflict pairs.  After those pairs are covered, prefer candidates
        that do not duplicate a selected conflict endpoint, then fill the
        requested budget.  The final exploratory slice is shuffled but also
        fills its allocation.
        """
        cap = max(0, int(cap))
        if cap == 0 or not ranked_candidates:
            return []

        candidates = list(dict.fromkeys(ranked_candidates))
        cap = min(cap, len(candidates))
        candidate_set = set(candidates)
        rank = {
            candidate: index
            for index, candidate in enumerate(candidates)
        }
        adjacency = defaultdict(set)
        for first, second in conflict_pairs or ():
            if (
                first == second
                or first not in candidate_set
                or second not in candidate_set
            ):
                continue
            adjacency[first].add(second)
            adjacency[second].add(first)

        fraction = min(1.0, max(0.0, float(exploration_fraction)))
        primary_target = max(
            1,
            min(cap, int(round(cap * (1.0 - fraction)))),
        )
        selected = []
        selected_set = set()
        remaining_adjacency = {
            candidate: set(adjacency.get(candidate, ()))
            for candidate in candidates
        }

        # Greedy vertex-cover approximation.  Lazy heap entries make degree
        # updates proportional to the affected conflict pairs instead of
        # rescanning every candidate for every slot.
        import heapq
        degree_heap = [
            (-len(remaining_adjacency[candidate]), rank[candidate], candidate)
            for candidate in candidates
            if remaining_adjacency[candidate]
        ]
        heapq.heapify(degree_heap)
        while len(selected) < primary_target and degree_heap:
            negative_degree, _, candidate = heapq.heappop(degree_heap)
            if candidate in selected_set:
                continue
            live_degree = len(remaining_adjacency[candidate])
            if -negative_degree != live_degree:
                heapq.heappush(
                    degree_heap,
                    (-live_degree, rank[candidate], candidate),
                )
                continue
            if live_degree == 0:
                break
            selected.append(candidate)
            selected_set.add(candidate)
            for neighbor in tuple(remaining_adjacency[candidate]):
                remaining_adjacency[neighbor].discard(candidate)
                heapq.heappush(
                    degree_heap,
                    (
                        -len(remaining_adjacency[neighbor]),
                        rank[neighbor],
                        neighbor,
                    ),
                )
            remaining_adjacency[candidate].clear()

        # Fill the deterministic part with nonredundant candidates first.
        # Edge-only offenders have no adjacency and therefore stay eligible
        # ahead of the opposite endpoint of an already covered conflict.
        if len(selected) < primary_target:
            nonredundant = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
                and not (
                    adjacency.get(candidate, set()) & selected_set
                )
            ]
            nonredundant_set = set(nonredundant)
            redundant = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
                and candidate not in nonredundant_set
            ]
            for candidate in nonredundant + redundant:
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) >= primary_target:
                    break

        if len(selected) < cap:
            exploration = [
                candidate
                for candidate in candidates
                if candidate not in selected_set
            ]
            rng.shuffle(exploration)
            selected.extend(exploration[:cap - len(selected)])

        return selected

    @staticmethod
    def _should_rip_for_stagnation(
        spatial_via_overuse: int,
        tail_threshold: int = 8,
        physical_cleanup_started: bool = False,
    ) -> bool:
        """Allow speculative rip-up only before staged physical cleanup."""
        return (
            not physical_cleanup_started
            and spatial_via_overuse <= max(0, int(tail_threshold))
        )

    @staticmethod
    def _scaled_via_keeper_rotation_threshold(
        base_threshold: int,
        routed_net_count: int,
        nets_per_step: int = 1024,
    ) -> int:
        """Scale a small-board via tail across independent net groups."""
        base = max(0, int(base_threshold))
        step = max(1, int(nets_per_step))
        groups = max(
            1,
            int(np.ceil(max(0, routed_net_count) / step)),
        )
        return base * groups

    def _via_keeper_rotation_threshold(self) -> int:
        """Return the live route's board-scaled rotation threshold."""
        routed_nets = sum(bool(path) for path in self.net_paths.values())
        return self._scaled_via_keeper_rotation_threshold(
            getattr(
                self.config,
                "via_keeper_rotation_overuse_threshold",
                8,
            ),
            routed_nets,
            getattr(
                self.config,
                "via_keeper_rotation_nets_per_step",
                1024,
            ),
        )

    def _clear_net_edge_tracking(self, net_id: str):
        """Clear edge-to-nets tracking for a net"""
        if net_id in self._net_to_edges:
            # Remove this net from all edge mappings
            for ei in self._net_to_edges[net_id]:
                self._edge_to_nets[ei].discard(net_id)
            del self._net_to_edges[net_id]

    def _select_physical_hotset(self) -> Set[str]:
        """Return a bounded, severity-ranked physical-conflict wave."""
        offenders = set(getattr(self, "_barrel_conflict_nets", ()))
        cap = self._physical_hotset_limit(
            int(getattr(self, "_last_barrel_conflict_count", 0)),
            max_cap=int(getattr(
                self.config, "physical_hotset_cap", 1024
            )),
            min_cap=int(getattr(
                self.config, "physical_hotset_min", 64
            )),
            conflicts_per_net=float(getattr(
                self.config,
                "physical_conflicts_per_hotset_net",
                50.0,
            )),
        )
        if len(offenders) <= cap:
            return offenders
        scores = getattr(self, "_physical_conflict_scores", {})
        ranked = sorted(
            offenders,
            key=lambda net_id: (
                -int(scores.get(net_id, 0)),
                str(net_id),
            ),
        )
        selected = set(ranked[:cap])
        logger.info(
            "[PHYSICAL-HOTSET] Selected %d/%d offenders "
            "(score range %d..%d)",
            len(selected),
            len(offenders),
            int(scores.get(ranked[0], 0)),
            int(scores.get(ranked[cap - 1], 0)),
        )
        return selected

    @staticmethod
    def _physical_hotset_limit(
        conflict_count: int,
        max_cap: int = 1024,
        min_cap: int = 64,
        conflicts_per_net: float = 50.0,
    ) -> int:
        """Shrink physical waves with the remaining conflict severity."""
        maximum = max(1, int(max_cap))
        minimum = min(maximum, max(1, int(min_cap)))
        scale = max(1.0, float(conflicts_per_net))
        severity_cap = int(np.ceil(max(0, conflict_count) / scale))
        return min(maximum, max(minimum, severity_cap))

    def _build_hotset(self, tasks: Dict[str, Tuple[int, int]], ripped: Optional[Set[str]] = None) -> Set[str]:
        """
        Build hotset: ONLY nets touching overused edges, with adaptive capping.
        Prevents thrashing by limiting hotset size based on actual overuse.
        Implements freeze-clean: nets clean for 3+ iterations are excluded from hotset.
        """
        if ripped is None:
            ripped = set()

        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)
        # History belongs in the Pathfinder routing cost, not in the
        # definition of a live offender.  Once an edge is no longer
        # oversubscribed, selecting every net that merely touches its retained
        # history wastes the bounded random hotset on clean nets and can make
        # full-memory runs diverge.  Rip up nets using resources that are
        # over capacity now; their next shortest-path search still sees the
        # complete historical cost field.
        over_idx = set(map(int, np.flatnonzero(over > 0)))
        via_pool_offenders = self._find_via_pool_offenders()
        path_node_scores = {
            net_id: int(score)
            for net_id, score in getattr(
                self, "_path_node_conflict_scores", {}
            ).items()
            if net_id in tasks and int(score) > 0
        }
        path_node_offenders = set(path_node_scores)
        path_node_overuse = 0
        if hasattr(self, "path_node_use"):
            path_node_overuse = int(np.maximum(
                0,
                np.asarray(self.path_node_use) - 1,
            ).sum())
        total_overuse_with_vias = self.accounting.compute_overuse(
            router_instance=self
        )[0]
        total_negotiated_overuse = (
            total_overuse_with_vias + path_node_overuse
        )
        cleanup_threshold = int(getattr(
            self.config,
            "portal_cleanup_edge_threshold",
            3,
        ))

        # Initialize clean iteration tracking
        if not hasattr(self, '_net_clean_iters'):
            self._net_clean_iters = {}

        freeze_after_clean = int(getattr(self.config, 'freeze_after_clean', 3))

        # No edge overuse can still leave an over-capacity via pool.
        if len(over_idx) == 0:
            unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}
            physical_offenders = (
                self._select_physical_hotset()
                if total_negotiated_overuse <= cleanup_threshold
                else set()
            )
            hotset = (
                unrouted
                | ripped
                | via_pool_offenders
                | physical_offenders
            )
            node_cap = min(
                int(self.config.hotset_cap),
                self._effective_history_hotset_cap(
                    total_negotiated_overuse
                ),
            )
            node_exploration_fraction = (
                self._hotset_exploration_fraction(
                    total_negotiated_overuse
                )
            )
            import random
            hotset.update(self._select_conflict_aware_hotset(
                sorted(
                    path_node_offenders,
                    key=lambda net_id: (
                        -path_node_scores[net_id],
                        str(net_id),
                    ),
                ),
                getattr(self, "_path_node_conflict_pairs", ()),
                node_cap,
                node_exploration_fraction,
                random.Random(42 + self.iteration),
            ))
            logger.info(
                f"[HOTSET] no-edge-overuse; unrouted={len(unrouted)} "
                f"ripped={len(ripped)} via_pool={len(via_pool_offenders)} "
                f"physical={len(physical_offenders)} "
                f"→ hotset={len(hotset)}"
            )
            self._last_hotset_size = len(hotset)
            self._last_hotset_cap = node_cap
            self._last_hotset_offender_count = len(
                path_node_offenders
            )
            self._last_hotset_exploration_fraction = (
                node_exploration_fraction
            )
            self._last_hotset_conflict_aware = True
            self._record_hotset_conflict_coverage(hotset)
            return hotset

        # OVERUSE EXISTS: collect nets touching overused edges using fast lookup
        offenders = set()
        for ei in over_idx:
            offenders.update(self._edge_to_nets.get(ei, set()))
        # Guided H/V routing can cross at a capacity-one lattice node without
        # sharing an edge. Negotiate those node-only offenders concurrently
        # with edge congestion instead of deferring them to final cleanup.
        offenders.update(path_node_offenders)

        # Update clean iteration counters
        for net_id in tasks.keys():
            if net_id in offenders:
                # Net is touching overused edges - reset counter
                self._net_clean_iters[net_id] = 0
            else:
                # Net is clean - increment counter
                self._net_clean_iters[net_id] = self._net_clean_iters.get(net_id, 0) + 1

        # Filter out frozen nets (clean for freeze_after_clean+ iterations)
        frozen_nets = {nid for nid in offenders if self._net_clean_iters.get(nid, 0) >= freeze_after_clean}
        offenders -= frozen_nets

        if frozen_nets:
            logger.debug(f"[FREEZE-CLEAN] Excluded {len(frozen_nets)} nets clean for {freeze_after_clean}+ iterations")

        # Add ripped nets
        offenders |= ripped
        offenders |= via_pool_offenders

        # Add unrouted nets (small priority, will be at end after sorting)
        unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}

        # Score offenders by total overuse they contribute
        scores = []
        for net_id in offenders:
            impact = float(path_node_scores.get(net_id, 0))
            if net_id in self._net_to_edges:
                impact += sum(
                    float(over[ei])
                    for ei in self._net_to_edges[net_id]
                    if ei in over_idx
                )
            scores.append((impact, net_id))

        # Add unrouted with low priority
        for net_id in unrouted:
            if net_id not in offenders:
                scores.append((0.0, net_id))

        # Sort by impact (highest first)
        scores.sort(reverse=True)

        # Scale with the complete edge/via + node residual. A small tail needs
        # cautious waves; severe monster-board congestion needs enough work
        # per pass to amortize full-graph accounting.
        total_overuse = sum(float(over[ei]) for ei in over_idx)

        # Preserve small tail waves, but do not apply the 100-net tail policy
        # to a monster route with tens of thousands of live node conflicts.
        # Exact physical offenders and unrouted nets still bypass this cap.
        base_target = self._effective_history_hotset_cap(
            total_negotiated_overuse
        )

        adaptive_cap = min(self.config.hotset_cap, base_target)

        # Severe congestion needs predominantly high-impact work. Retain a
        # bounded random fraction to break phase-locking, then select an
        # independent set of current path-node conflicts so both sides do not
        # move together and exchange ownership.
        import random
        rng = random.Random(42 + self.iteration)

        # Cooldown: exclude nets rerouted in previous iteration (prevents immediate re-routing)
        if not hasattr(self, '_last_reroute_iter'):
            self._last_reroute_iter = {}

        ranked_with_cooldown = [
            net_id
            for _, net_id in scores
            if (
                self.iteration
                - self._last_reroute_iter.get(net_id, -999)
                > 1
            )
        ]
        exploration_fraction = self._hotset_exploration_fraction(
            total_negotiated_overuse
        )
        hotset_with_cooldown = self._select_conflict_aware_hotset(
            ranked_with_cooldown,
            getattr(self, "_path_node_conflict_pairs", ()),
            adaptive_cap,
            exploration_fraction,
            rng,
        )

        # Update last reroute iteration for selected nets
        for nid in hotset_with_cooldown:
            self._last_reroute_iter[nid] = self.iteration

        hotset = set(hotset_with_cooldown)
        raw_overuse_edges = int(np.count_nonzero(over > 0))
        if (
            raw_overuse_edges <= cleanup_threshold
            and total_negotiated_overuse <= cleanup_threshold
        ):
            physical_offenders = self._select_physical_hotset()
            # The adaptive cap is for ordinary edge congestion. Once the graph
            # tail is clean enough for physical cleanup, exact shorts bypass
            # that cap so one-sided repair can move every selected component.
            hotset.update(physical_offenders)
        # A failed full-graph search has no committed edges and therefore no
        # congestion score. It must bypass both caps and cooldowns or it can
        # remain unrouted indefinitely.
        hotset.update(unrouted)

        unique_frac = len(hotset - getattr(self, '_prev_hotset', set())) / max(1, len(hotset))
        self._prev_hotset = hotset.copy()
        self._last_hotset_size = len(hotset)
        self._last_hotset_cap = adaptive_cap
        self._last_hotset_offender_count = len(offenders)
        self._last_hotset_exploration_fraction = (
            exploration_fraction
        )
        self._last_hotset_conflict_aware = True
        self._record_hotset_conflict_coverage(hotset)

        logger.info(f"[HOTSET] overuse_edges={len(over_idx)} total_overuse={int(total_overuse)}, "
                    f"offenders={len(offenders)}, cap={adaptive_cap} → hotset={len(hotset)}/{len(tasks)} "
                    f"(explore={exploration_fraction:.0%}, "
                    f"conflict-aware, unique={unique_frac:.1%}, "
                    f"pair-cover="
                    f"{self._last_hotset_conflict_pair_coverage_fraction:.1%})")

        return hotset

    def _find_via_pool_offenders(self) -> Set[str]:
        """Return only excess peers on each over-capacity via resource."""
        if not hasattr(self, "via_col_use") and not hasattr(
            self, "via_seg_use"
        ):
            return set()

        def over_mask(use, capacity):
            mask = use > capacity
            if hasattr(mask, "get"):
                if not bool(cp.any(mask)):
                    return None
                return mask.get()
            return mask if np.any(mask) else None

        col_over = None
        if hasattr(self, "via_col_use") and hasattr(self, "via_col_cap"):
            col_over = over_mask(self.via_col_use, self.via_col_cap)

        seg_over = None
        if hasattr(self, "via_seg_use") and hasattr(self, "via_seg_cap"):
            seg_over = over_mask(self.via_seg_use, self.via_seg_cap)

        if col_over is None and seg_over is None:
            self._via_pool_conflict_nets = set()
            self._via_pool_keepers = {}
            self._via_pool_member_state = {}
            self._via_pool_keeper_stagnation = {}
            return set()

        members = defaultdict(set)
        for net_id, path in self.net_paths.items():
            if not path or len(path) < 2:
                continue
            net_resources = set()
            for u, v in zip(path, path[1:]):
                xu, yu, zu = self.lattice.idx_to_coord(u)
                xv, yv, zv = self.lattice.idx_to_coord(v)
                if xu != xv or yu != yv or zu == zv:
                    continue
                if col_over is not None and col_over[xu, yu]:
                    net_resources.add(("col", xu, yu))
                if seg_over is not None:
                    z_lo, z_hi = sorted((zu, zv))
                    z_lo = max(1, min(z_lo, self._Nz - 2))
                    z_hi = max(1, min(z_hi, self._Nz - 2))
                    for z in range(z_lo, z_hi):
                        seg_idx = z - 1
                        if (
                            0 <= seg_idx < self._segZ
                            and seg_over[xu, yu, seg_idx]
                        ):
                            net_resources.add(
                                ("seg", xu, yu, seg_idx)
                            )
            for resource in net_resources:
                members[resource].add(net_id)

        old_keepers = getattr(self, "_via_pool_keepers", {})
        old_member_state = getattr(
            self, "_via_pool_member_state", {}
        )
        old_stagnation = getattr(
            self, "_via_pool_keeper_stagnation", {}
        )
        rotation_allowed = (
            self._spatial_via_overuse_total()
            <= self._via_keeper_rotation_threshold()
        )
        new_keepers = {}
        new_member_state = {}
        new_stagnation = {}
        offenders = set()

        col_cap = getattr(self, "via_col_cap", None)
        if hasattr(col_cap, "get"):
            col_cap = col_cap.get()
        seg_cap = getattr(self, "via_seg_cap", None)
        if hasattr(seg_cap, "get"):
            seg_cap = seg_cap.get()
        col_use = getattr(self, "via_col_use", None)
        if hasattr(col_use, "get"):
            col_use = col_use.get()
        seg_use = getattr(self, "via_seg_use", None)
        if hasattr(seg_use, "get"):
            seg_use = seg_use.get()

        for resource, resource_members in sorted(members.items()):
            kind, x_idx, y_idx, *tail = resource
            if kind == "col":
                capacity = int(col_cap[x_idx, y_idx])
                observed = int(col_use[x_idx, y_idx])
            else:
                seg_idx = tail[0]
                capacity = int(seg_cap[x_idx, y_idx, seg_idx])
                observed = int(seg_use[x_idx, y_idx, seg_idx])

            capacity = max(0, capacity)
            member_state = frozenset(resource_members)
            new_member_state[resource] = member_state
            unchanged = (
                old_member_state.get(resource) == member_state
            )
            stagnant = (
                int(old_stagnation.get(resource, 0)) + 1
                if unchanged else 0
            )
            previous = [
                net_id
                for net_id in old_keepers.get(resource, ())
                if net_id in resource_members
            ][:capacity]
            remaining = sorted(resource_members - set(previous))
            keepers = tuple(
                previous + remaining[:max(0, capacity - len(previous))]
            )

            # A stable keeper prevents peers from exchanging ownership every
            # pass, but it can starve the cleanup when the chosen excess net
            # has no viable alternative. After two unchanged attempts, rotate
            # the protected window so a former keeper gets a chance to move.
            if (
                rotation_allowed
                and
                stagnant >= 2
                and 0 < capacity < len(resource_members)
                and previous
            ):
                ordered = sorted(resource_members)
                start = (ordered.index(previous[0]) + 1) % len(ordered)
                keepers = tuple(
                    ordered[(start + offset) % len(ordered)]
                    for offset in range(capacity)
                )
                stagnant = 0
                logger.info(
                    "[VIA-POOL] Rotated %s keepers at (%d,%d)",
                    kind,
                    x_idx,
                    y_idx,
                )

            new_stagnation[resource] = stagnant
            new_keepers[resource] = keepers
            offenders.update(resource_members - set(keepers))

            # Repeated disconnected barrels from one net can make observed
            # usage exceed the number of electrical owners. That net must move
            # even if it would otherwise be the sole keeper.
            if observed > len(resource_members):
                offenders.update(resource_members)

        self._via_pool_keepers = new_keepers
        self._via_pool_member_state = new_member_state
        self._via_pool_keeper_stagnation = new_stagnation
        self._via_pool_conflict_nets = set().union(
            *members.values()
        ) if members else set()
        return offenders

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

    def _log_per_layer_congestion(self, over: np.ndarray):
        """Log overuse breakdown by layer (GPU-accelerated version)"""
        import time
        start_time = time.time()

        # Use edge_layer array if available (pre-computed), otherwise compute from graph
        if hasattr(self.graph, 'edge_layer'):
            edge_layer = self.graph.edge_layer.get() if hasattr(self.graph.edge_layer, 'get') else self.graph.edge_layer
            logger.debug(f"[LAYER-DIAG] Using pre-computed edge_layer array ({len(edge_layer)} edges)")
        else:
            # Fallback: compute layers from nodes (slower but works)
            logger.debug(f"[LAYER-DIAG] Computing edge layers from graph (fallback mode)")
            indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
            indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices
            plane_size = self.lattice.Nx * self.lattice.Ny

            # Vectorized: find source nodes for all edges
            edge_indices = np.arange(len(over), dtype=np.int32)
            src_nodes = np.searchsorted(indptr, edge_indices, side='right') - 1
            dst_nodes = indices[edge_indices]

            src_layers = src_nodes // plane_size
            dst_layers = dst_nodes // plane_size
            edge_layer = src_layers  # Use source layer for classification
            logger.debug(f"[LAYER-DIAG] Computed layers for {len(edge_layer)} edges in {time.time()-start_time:.2f}s")

        # GPU-accelerated aggregation
        layer_count = self.config.layer_count
        overuse_horiz = {}

        # Filter to overused edges only
        overused_mask = over > 0
        num_overused = int(np.sum(overused_mask))

        if num_overused == 0:
            logger.debug(f"[LAYER-DIAG] No overused edges, skipping")
            return {}

        logger.debug(f"[LAYER-DIAG] Processing {num_overused}/{len(over)} overused edges")

        overused_values = over[overused_mask]
        overused_layers = edge_layer[overused_mask]

        # Sum by layer (vectorized)
        for z in range(1, layer_count + 1):
            layer_mask = overused_layers == z
            if np.any(layer_mask):
                overuse_horiz[z] = float(np.sum(overused_values[layer_mask]))

        elapsed = time.time() - start_time
        logger.debug(f"[LAYER-DIAG] Completed in {elapsed:.3f}s (vectorized)")

        # Log horizontal overuse by layer
        total_horiz = sum(overuse_horiz.values())
        if total_horiz > 0:
            logger.info(f"[LAYER-CONGESTION] Horizontal overuse by layer:")
            for z in sorted(overuse_horiz.keys()):
                if overuse_horiz[z] > 0:
                    pct = (overuse_horiz[z] / total_horiz) * 100
                    logger.info(f"  Layer {z:2d}: {overuse_horiz[z]:7.1f} ({pct:5.1f}%)")

        return overuse_horiz  # Return for layer balancing

    def _update_layer_bias(self, overuse_by_layer: dict, layer_bias: dict) -> dict:
        """
        Update layer bias based on overuse distribution to encourage load balancing.
        Hot layers get positive bias (higher cost), cool layers get negative bias (lower cost).
        Uses EMA to smooth changes and prevent whiplash.
        """
        if not overuse_by_layer or sum(overuse_by_layer.values()) == 0:
            return layer_bias

        # Calculate mean overuse (ignoring zero layers to avoid dilution)
        nonzero_overuse = [v for v in overuse_by_layer.values() if v > 0]
        if not nonzero_overuse:
            return layer_bias

        mu = sum(nonzero_overuse) / len(nonzero_overuse) + 1e-6

        # Calculate normalized error for each layer (positive = hotter than average)
        for z in overuse_by_layer.keys():
            err = (overuse_by_layer[z] / mu) - 1.0
            # EMA update with small gain (0.02) to prevent oscillation
            # Multiply by small factor (0.05) for gentle nudging
            old_bias = layer_bias.get(z, 0.0)
            new_bias = 0.9 * old_bias + 0.1 * (0.05 * err)
            layer_bias[z] = new_bias

        # Log top 5 biases for debugging
        top_biases = sorted(layer_bias.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        if any(abs(bias) > 0.01 for _, bias in top_biases):
            bias_str = ", ".join([f"L{z}:{bias:+.3f}" for z, bias in top_biases])
            logger.info(f"[LAYER-BIAS] Top biases: {bias_str}")

        return layer_bias

    def _apply_layer_bias_to_costs(self, layer_bias: dict):
        """Apply layer bias to edge costs (only horizontal edges on each layer)"""
        if not layer_bias or not hasattr(self, '_layer_edges'):
            return

        # Get current costs
        total_cost = self.accounting.total_cost
        if self.accounting.use_gpu:
            total_cost_cpu = total_cost.get()
        else:
            total_cost_cpu = total_cost

        # Apply bias to each layer's horizontal edges
        for z, bias in layer_bias.items():
            if abs(bias) < 0.001:  # Skip negligible biases
                continue
            if z in self._layer_edges:
                edge_indices = self._layer_edges[z]
                total_cost_cpu[edge_indices] += bias

        # Update GPU if needed
        if self.accounting.use_gpu:
            self.accounting.total_cost[:] = cp.asarray(total_cost_cpu)

    def _rank_stagnation_offenders(
        self,
        over: np.ndarray,
    ) -> List[Tuple[float, str]]:
        """Rank live offenders by the complete negotiated objective."""
        canonical_mask = self._canonical_edge_resource_mask()
        over_idx = set(map(
            int,
            np.flatnonzero((over > 0) & canonical_mask),
        ))
        _, _, measured_node_scores = (
            self._detect_path_node_conflicts()
        )
        self._path_node_conflict_scores = dict(measured_node_scores)

        candidates = set(measured_node_scores)
        for edge_idx in over_idx:
            candidates.update(self._edge_to_nets.get(edge_idx, ()))

        scores = []
        for net_id in candidates:
            if (
                not self.net_paths.get(net_id)
                or net_id in self.locked_nets
            ):
                continue
            impact = float(measured_node_scores.get(net_id, 0))
            if net_id in self._net_to_edges:
                impact += sum(
                    float(over[edge_idx])
                    for edge_idx in self._net_to_edges[net_id]
                    if edge_idx in over_idx
                )
            if impact > 0:
                scores.append((impact, net_id))
        return sorted(
            scores,
            key=lambda item: (-item[0], str(item[1])),
        )

    def _rip_top_k_offenders(self, k=20) -> Set[str]:
        """
        Rip only the worst 16-24 nets to break stagnation (not the world).
        Respect locked nets - don't rip unless they touch new overuse.
        Returns the set of ripped net IDs.
        """
        present = self.accounting.present.get() if self.accounting.use_gpu else self.accounting.present
        cap = self.accounting.capacity.get() if self.accounting.use_gpu else self.accounting.capacity
        over = np.maximum(0, present - cap)
        scores = self._rank_stagnation_offenders(over)
        victims = self._select_stagnation_victims(scores, k)

        for net_id in victims:
            if self.net_paths.get(net_id) and net_id in self._net_to_edges:
                old_path = self.net_paths[net_id]
                # Use cached edges for efficiency
                self.accounting.clear_path(self._net_to_edges[net_id])
                self._clear_via_barrel_ownership_for_path(
                    net_id, old_path
                )
                self._clear_path_node_use(old_path)
                self._clear_escape_occupancy(net_id)
                self.net_paths[net_id] = []
                self.net_selected_portals.pop(net_id, None)
                # Clear edge tracking for ripped nets
                self._clear_net_edge_tracking(net_id)
                # Reset clean streak so they can't immediately lock again
                self.net_clean_streak[net_id] = 0

        logger.info(f"[STAGNATION] Ripped {len(victims)} nets (locked={len(self.locked_nets)} preserved)")
        return victims

    def _select_stagnation_victims(
        self,
        scores: Sequence[Tuple[float, str]],
        k: int,
    ) -> Set[str]:
        """Select the next untried recovery wave around one retained best."""
        prior_victims = getattr(
            self, "_stagnation_victim_history", set()
        )
        untried = [
            net_id for _, net_id in scores
            if net_id not in prior_victims
        ]
        selected = list(untried[:k])
        if len(selected) < k:
            # Finish the tail of the current cycle before reusing the
            # highest-ranked victims at the start of the next one.
            prior_victims = set()
            for _, net_id in scores:
                if net_id in selected:
                    continue
                selected.append(net_id)
                if len(selected) >= k:
                    break
        victims = set(selected)
        self._stagnation_victim_history = (
            prior_victims | victims
        )
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
        if getattr(self.graph, "edge_kind", None) is not None:
            # CSR finalization has already computed this classification.  Do
            # not download and rescan the multi-gigabyte CSR on GPU builds.
            self._via_edges = self.graph.edge_kind.astype(np.bool_, copy=True)
            logger.info(
                f"Identified {int(self._via_edges.sum())} via edges "
                "from cached edge metadata"
            )
            return

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

    def _build_via_edge_metadata(self):
        """
        Precompute via edge metadata for vectorized penalty application.

        Keeps metadata on GPU if available for zero-copy kernel execution.
        """
        import time
        t0 = time.perf_counter()
        use_via_gpu = self.config.use_gpu and GPU_AVAILABLE

        if use_via_gpu:
            # Build directly from the device-resident CSR.  Pulling indices
            # back to host and uploading four derived arrays temporarily
            # duplicates several gigabytes on large backplanes.
            via_edge_indices = cp.flatnonzero(
                self.graph.edge_kind_gpu
            ).astype(cp.int32)
            num_via_edges = int(via_edge_indices.size)
            if num_via_edges == 0:
                self._via_edge_metadata = None
                return

            u_indices = (
                cp.searchsorted(
                    self.graph.indptr, via_edge_indices, side='right'
                ) - 1
            ).astype(cp.int32)
            v_indices = self.graph.indices[via_edge_indices]
            plane_size = self.lattice.x_steps * self.lattice.y_steps

            xu = (u_indices % plane_size) % self.lattice.x_steps
            yu = (u_indices % plane_size) // self.lattice.x_steps
            zu = u_indices // plane_size
            zv = v_indices // plane_size

            via_xy_coords = cp.stack([xu, yu], axis=1).astype(cp.int32)
            z_lo = cp.clip(
                cp.minimum(zu, zv), 1, self.lattice.layers - 2
            ).astype(cp.int32)
            z_hi = cp.clip(
                cp.maximum(zu, zv), 1, self.lattice.layers - 2
            ).astype(cp.int32)

            self._via_edge_metadata = {
                'indices': via_edge_indices,
                'xy_coords': via_xy_coords,
                'z_lo': z_lo,
                'z_hi': z_hi,
            }
            logger.info(
                f"[VIA-METADATA] Built metadata for {num_via_edges} "
                f"via edges on GPU in {time.perf_counter() - t0:.3f}s"
            )
            return

        # Get via edge indices
        via_edge_indices = np.where(self._via_edges)[0]
        num_via_edges = len(via_edge_indices)

        if num_via_edges == 0:
            self._via_edge_metadata = None
            return

        # Get graph data
        indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
        indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices

        # Precompute u (source node) for each via edge using searchsorted
        u_indices = np.searchsorted(indptr, via_edge_indices, side='right') - 1

        # Get v (destination node) for each via edge
        v_indices = indices[via_edge_indices]

        # Convert to coordinates (vectorized)
        plane_size = self.lattice.x_steps * self.lattice.y_steps

        # u coordinates
        xu = (u_indices % plane_size) % self.lattice.x_steps
        yu = (u_indices % plane_size) // self.lattice.x_steps
        zu = u_indices // plane_size

        # v coordinates
        xv = (v_indices % plane_size) % self.lattice.x_steps
        yv = (v_indices % plane_size) // self.lattice.x_steps
        zv = v_indices // plane_size

        # For via edges, x,y should be same (sanity check in debug mode)
        # Store just one (x,y) coordinate per via edge
        via_xy_coords = np.stack([xu, yu], axis=1).astype(np.int32)

        # Store z ranges (lo, hi) for each via edge
        z_lo = np.minimum(zu, zv)
        z_hi = np.maximum(zu, zv)

        # Clamp z values to valid routing layers (1..Nz-2)
        z_lo = np.clip(z_lo, 1, self.lattice.layers - 2)
        z_hi = np.clip(z_hi, 1, self.lattice.layers - 2)

        self._via_edge_metadata = {
            'indices': via_edge_indices.astype(np.int32),
            'xy_coords': via_xy_coords,
            'z_lo': z_lo.astype(np.int32),
            'z_hi': z_hi.astype(np.int32),
        }
        logger.info(f"[VIA-METADATA] Built metadata for {num_via_edges} via edges on CPU in {time.perf_counter() - t0:.3f}s")

    def _path_to_directed_edges(
        self, node_path: List[int]
    ) -> List[int]:
        """Map traversal hops to directed CSR arc ids."""
        if len(node_path) < 2:
            return []

        if self._indptr_cpu is None:
            self._indptr_cpu = (
                self.graph.indptr.get()
                if hasattr(self.graph.indptr, "get")
                else np.asarray(self.graph.indptr)
            )
            self._indices_cpu = (
                self.graph.indices.get()
                if hasattr(self.graph.indices, "get")
                else np.asarray(self.graph.indices)
            )

        indptr = self._indptr_cpu
        indices = self._indices_cpu
        nodes = np.asarray(node_path, dtype=np.int64)
        sources = nodes[:-1]
        targets = nodes[1:]
        row_start = indptr[sources]
        row_end = indptr[sources + 1]
        row_len = row_end - row_start
        max_row = int(row_len.max(initial=0))

        if max_row == 0:
            raise ValueError(
                f"Path starts with a node that has no graph edges: "
                f"{int(sources[0])}→{int(targets[0])}"
            )

        offsets = np.arange(max_row, dtype=np.int64)
        edge_candidates = row_start[:, None] + offsets[None, :]
        valid = offsets[None, :] < row_len[:, None]
        safe_candidates = np.minimum(edge_candidates, len(indices) - 1)
        neighbors = np.where(valid, indices[safe_candidates], -1)
        matches = neighbors == targets[:, None]
        has_match = matches.any(axis=1)

        if not np.all(has_match):
            hop = int(np.flatnonzero(~has_match)[0])
            raise ValueError(
                f"Path hop is not a graph edge at offset {hop}: "
                f"{int(sources[hop])}→{int(targets[hop])}"
            )

        edge_column = matches.argmax(axis=1)
        return (row_start + edge_column).astype(np.int64).tolist()

    def _path_to_edges(self, node_path: List[int]) -> List[int]:
        """Map a path to its undirected physical-resource arc ids.

        The CSR has one arc per travel direction, but a copper segment or via
        is one shared physical resource. Reserving both arcs makes opposite
        traversals collide during ordinary PathFinder negotiation instead of
        escaping to the later path-node cleanup phase.
        """
        forward = self._path_to_directed_edges(node_path)
        reverse = self._path_to_directed_edges(
            list(reversed(node_path))
        )
        return forward + reverse

    def _accumulate_node_conflict_history(self, nodes) -> None:
        """Learn each physically shorted node at most once per iteration."""
        if not hasattr(self, "node_conflict_history"):
            return
        iteration = int(getattr(self, "iteration", -1))
        if isinstance(nodes, set):
            nodes = tuple(nodes)
        nodes = np.unique(np.asarray(nodes, dtype=np.int64))
        if nodes.size == 0:
            return
        if getattr(
            self, "_node_history_iteration", None
        ) != iteration:
            self._node_history_iteration = iteration
            self._node_history_nodes = set()
        learned = getattr(self, "_node_history_nodes", set())
        nodes = np.asarray(
            [node for node in nodes if int(node) not in learned],
            dtype=np.int64,
        )
        if nodes.size == 0:
            return
        increment = np.float32(getattr(
            self.config, "node_history_increment", 1.0
        ))
        self.node_conflict_history[nodes] += increment
        if self.node_conflict_history_gpu is not None:
            nodes_gpu = cp.asarray(nodes, dtype=cp.int32)
            self.node_conflict_history_gpu[nodes_gpu] = cp.asarray(
                self.node_conflict_history[nodes],
                dtype=cp.float32,
            )
        learned.update(map(int, nodes))
        self._node_history_nodes = learned

    def _edge_index_for_hop(
        self, source: int, target: int
    ) -> Optional[int]:
        """Return one directed CSR edge index, or None for a non-edge."""
        if self._indptr_cpu is None:
            self._indptr_cpu = (
                self.graph.indptr.get()
                if hasattr(self.graph.indptr, "get")
                else np.asarray(self.graph.indptr)
            )
            self._indices_cpu = (
                self.graph.indices.get()
                if hasattr(self.graph.indices, "get")
                else np.asarray(self.graph.indices)
            )
        start = int(self._indptr_cpu[source])
        end = int(self._indptr_cpu[source + 1])
        row = self._indices_cpu[start:end]
        matches = np.flatnonzero(row == target)
        if matches.size == 0:
            return None
        return start + int(matches[0])

    def _iter_planar_segments_in_box(
        self, layer: int, x_lo: int, x_hi: int, y_lo: int, y_hi: int
    ):
        """Yield every graph-planar segment in an inclusive XY box."""
        for axis in self.lattice.get_allowed_axes(layer):
            if axis == "h":
                yield from (
                    (x_idx, y_idx, x_idx + 1, y_idx)
                    for y_idx in range(y_lo, y_hi + 1)
                    for x_idx in range(x_lo, x_hi)
                )
            else:
                yield from (
                    (x_idx, y_idx, x_idx, y_idx + 1)
                    for x_idx in range(x_lo, x_hi + 1)
                    for y_idx in range(y_lo, y_hi)
                )

    def _portal_conflicting_graph_edges(
        self, portal: Portal, entry_layer: int
    ) -> np.ndarray:
        """Return exact graph edges violating one terminal-via envelope."""
        cache = getattr(
            self, "_portal_conflicting_edges_cache", None
        )
        if cache is None:
            cache = {}
            self._portal_conflicting_edges_cache = cache
        cache_key = (id(portal), int(entry_layer))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        self._ensure_edge_src_map()
        center = self.escape_planner._portal_world(portal)
        pitch = float(self.lattice.geom.pitch)
        via_track_limit = max(
            0.5 * float(self.config.via_diameter)
            + 0.5 * float(self.config.track_width)
            + float(self.config.clearance),
            0.5 * float(self.config.via_drill)
            + 0.5 * float(self.config.track_width)
            + float(getattr(self.config, "hole_clearance", 0.0)),
        )
        via_via_limit = max(
            float(self.config.via_diameter)
            + float(self.config.clearance),
            float(self.config.via_drill)
            + float(getattr(
                self.config, "min_hole_to_hole", 0.0
            )),
        )
        search_steps = int(np.ceil(
            max(via_track_limit, via_via_limit) / pitch
        )) + 1
        x_lo = max(0, portal.x_idx - search_steps)
        x_hi = min(
            self.lattice.x_steps - 1,
            portal.x_idx + search_steps,
        )
        y_lo = max(0, portal.y_idx - search_steps)
        y_hi = min(
            self.lattice.y_steps - 1,
            portal.y_idx + search_steps,
        )
        z_lo, z_hi = sorted((
            int(portal.pad_layer), int(entry_layer)
        ))
        edges = set()

        for layer in range(z_lo, z_hi + 1):
            if not (
                self.lattice.layers > 2
                and layer in (0, self.lattice.layers - 1)
            ):
                for x0, y0, x1, y1 in (
                    self._iter_planar_segments_in_box(
                        layer, x_lo, x_hi, y_lo, y_hi
                    )
                ):
                    start = self.lattice.geom.lattice_to_world(
                        x0, y0
                    )
                    end = self.lattice.geom.lattice_to_world(x1, y1)
                    if (
                        self._point_segment_distance(
                            center, start, end
                        )
                        >= via_track_limit - 1e-9
                    ):
                        continue
                    source = self.lattice.node_idx(
                        x0, y0, layer
                    )
                    target = self.lattice.node_idx(
                        x1, y1, layer
                    )
                    for u, v in (
                        (source, target), (target, source)
                    ):
                        edge = self._edge_index_for_hop(u, v)
                        if edge is not None:
                            edges.add(edge)

            for x_idx in range(x_lo, x_hi + 1):
                for y_idx in range(y_lo, y_hi + 1):
                    world = self.lattice.geom.lattice_to_world(
                        x_idx, y_idx
                    )
                    if (
                        float(np.hypot(
                            world[0] - center[0],
                            world[1] - center[1],
                        ))
                        >= via_via_limit - 1e-9
                    ):
                        continue
                    node = self.lattice.node_idx(
                        x_idx, y_idx, layer
                    )
                    start = int(self._indptr_cpu[node])
                    end = int(self._indptr_cpu[node + 1])
                    for edge in range(start, end):
                        target = int(self._indices_cpu[edge])
                        target_layer = (
                            target
                            // (
                                self.lattice.x_steps
                                * self.lattice.y_steps
                            )
                        )
                        if target_layer == layer:
                            continue
                        edge_lo, edge_hi = sorted((
                            layer, target_layer
                        ))
                        if edge_lo <= z_hi and edge_hi >= z_lo:
                            edges.add(edge)
                            reverse = self._edge_index_for_hop(
                                target, node
                            )
                            if reverse is not None:
                                edges.add(reverse)

        result = np.asarray(sorted(edges), dtype=np.int32)
        cache[cache_key] = result
        return result

    def _rebuild_portal_cleanup_edge_owners(self) -> None:
        """Index fixed terminal envelopes by their exact graph edges."""
        members = defaultdict(set)
        for net_name, selected in self.net_selected_portals.items():
            if not self.net_paths.get(net_name):
                continue
            net_id = self._get_net_id(net_name)
            layers = self.net_portal_layers.get(net_name, ())
            for portal, entry_layer in zip(selected, layers):
                if portal is None:
                    continue
                for edge in self._portal_conflicting_graph_edges(
                    portal, int(entry_layer)
                ):
                    members[int(edge)].add(net_id)
        self._portal_cleanup_edge_members = members
        self._portal_cleanup_foreign_cache = {}
        logger.info(
            "[PORTAL-CLEANUP] Indexed %d exact graph edges around "
            "fixed terminal vias",
            len(members),
        )

    @staticmethod
    def _should_run_one_sided_cleanup(
        physical_conflicts: int,
        overused_edges: int,
        already_active: bool,
        edge_threshold: int = 0,
        overuse_total: Optional[float] = None,
    ) -> bool:
        """Run only while ordinary graph edges remain below the tail gate."""
        if overuse_total is None:
            overuse_total = overused_edges
        return (
            physical_conflicts > 0
            and overused_edges <= max(0, int(edge_threshold))
            and overuse_total <= max(0, int(edge_threshold))
        )

    def _portal_cleanup_movable_components(
        self, portal_grid_pairs, escape_pairs=(), exact_pairs=()
    ) -> Set[str]:
        """Choose a deterministic independent set of physical-conflict nets.

        Rerouting every peer in a large connected component moves both ends of
        most shorts and simply migrates them. Rerouting only one peer makes a
        connector-sized component progress serially. A greedy maximal
        independent set gives a broad wave in which no two current conflict
        peers can move together. Prefer the least-tried endpoint first, then
        high-degree nets, so a stable pair alternates sides instead of starving
        one endpoint forever.
        """
        adjacency = defaultdict(set)
        for identity, victim, _kind in portal_grid_pairs:
            owner = identity[0]
            if owner == victim:
                continue
            adjacency[owner].add(victim)
            adjacency[victim].add(owner)
        for first, second in escape_pairs:
            first_net = first[0]
            second_net = second[0]
            if first_net == second_net:
                continue
            adjacency[first_net].add(second_net)
            adjacency[second_net].add(first_net)
        for first_net, second_net in exact_pairs:
            if first_net == second_net:
                continue
            adjacency[first_net].add(second_net)
            adjacency[second_net].add(first_net)

        movable = set()
        eligible = set(adjacency)
        while eligible:
            net_name = min(
                eligible,
                key=lambda candidate: (
                    self._portal_cleanup_move_counts[candidate],
                    -len(adjacency[candidate]),
                    candidate,
                ),
            )
            movable.add(net_name)
            eligible.discard(net_name)
            eligible.difference_update(adjacency[net_name])
        for net_name in movable:
            self._portal_cleanup_move_counts[net_name] += 1
        return movable

    def _portal_cleanup_foreign_edges(
        self, current_net: str
    ) -> np.ndarray:
        """Return exact terminal-clearance edges owned by other nets."""
        cache = getattr(
            self, "_portal_cleanup_foreign_cache", {}
        )
        cached = cache.get(current_net)
        if cached is not None:
            return cached
        current_id = self._get_net_id(current_net)
        result = np.fromiter(
            (
                edge
                for edge, members in getattr(
                    self, "_portal_cleanup_edge_members", {}
                ).items()
                if any(owner != current_id for owner in members)
            ),
            dtype=np.int32,
        )
        cache[current_net] = result
        self._portal_cleanup_foreign_cache = cache
        return result

    def _detect_portal_grid_conflicts(self):
        """Audit explicit terminal vias against committed lattice copper.

        Terminal vias may be off-grid and are intentionally contracted out
        of the routing graph. Check their real emitted center, diameter,
        drill, and layer span against graph tracks and graph-via barrels.
        """
        if self.escape_planner is None:
            return set(), set(), set(), set(), set()

        self._ensure_edge_src_map()
        edge_owners = defaultdict(set)
        node_owners = defaultdict(set)
        portal_keys = {}
        pitch = float(self.lattice.geom.pitch)
        via_track_limit = max(
            0.5 * float(self.config.via_diameter)
            + 0.5 * float(self.config.track_width)
            + float(self.config.clearance),
            0.5 * float(self.config.via_drill)
            + 0.5 * float(self.config.track_width)
            + float(getattr(self.config, "hole_clearance", 0.0)),
        )
        via_via_limit = max(
            float(self.config.via_diameter)
            + float(self.config.clearance),
            float(self.config.via_drill)
            + float(getattr(
                self.config, "min_hole_to_hole", 0.0
            )),
        )
        search_steps = int(np.ceil(
            max(via_track_limit, via_via_limit) / pitch
        )) + 1

        for net_name, selected in self.net_selected_portals.items():
            if not self.net_paths.get(net_name):
                continue
            pad_ids = self.net_pad_ids.get(net_name, ())
            layers = self.net_portal_layers.get(net_name, ())
            for pad_id, portal, entry_layer in zip(
                pad_ids, selected, layers
            ):
                if portal is None:
                    continue
                identity = (
                    net_name,
                    pad_id,
                    portal.x_idx,
                    portal.y_idx,
                )
                portal_keys[identity] = (
                    pad_id,
                    portal.x_idx,
                    portal.y_idx,
                    int(entry_layer),
                )
                center = self.escape_planner._portal_world(portal)
                z_lo, z_hi = sorted((
                    int(portal.pad_layer), int(entry_layer)
                ))
                x_lo = max(0, portal.x_idx - search_steps)
                x_hi = min(
                    self.lattice.x_steps - 1,
                    portal.x_idx + search_steps,
                )
                y_lo = max(0, portal.y_idx - search_steps)
                y_hi = min(
                    self.lattice.y_steps - 1,
                    portal.y_idx + search_steps,
                )

                for layer in range(z_lo, z_hi + 1):
                    for x_idx in range(x_lo, x_hi + 1):
                        for y_idx in range(y_lo, y_hi + 1):
                            world = self.lattice.geom.lattice_to_world(
                                x_idx, y_idx
                            )
                            if (
                                float(np.hypot(
                                    world[0] - center[0],
                                    world[1] - center[1],
                                ))
                                < via_via_limit - 1e-9
                            ):
                                node_owners[
                                    self.lattice.node_idx(
                                        x_idx, y_idx, layer
                                    )
                                ].add(identity)

                    if (
                        self.lattice.layers > 2
                        and layer in (0, self.lattice.layers - 1)
                    ):
                        continue
                    for x0, y0, x1, y1 in (
                        self._iter_planar_segments_in_box(
                            layer, x_lo, x_hi, y_lo, y_hi
                        )
                    ):
                        start_world = (
                            self.lattice.geom.lattice_to_world(x0, y0)
                        )
                        end_world = (
                            self.lattice.geom.lattice_to_world(x1, y1)
                        )
                        if (
                            self._point_segment_distance(
                                center, start_world, end_world
                            )
                            >= via_track_limit - 1e-9
                        ):
                            continue
                        source = self.lattice.node_idx(x0, y0, layer)
                        target = self.lattice.node_idx(x1, y1, layer)
                        for u, v in (
                            (source, target), (target, source)
                        ):
                            edge = self._edge_index_for_hop(u, v)
                            if edge is not None:
                                edge_owners[edge].add(identity)

        pairs = set()
        owner_nets = set()
        victim_nets = set()
        history_nodes = set()
        involved_portals = set()

        for edge, identities in edge_owners.items():
            victims = self._edge_to_nets.get(edge, ())
            if not victims:
                continue
            source = int(self._edge_src[edge])
            target = int(self.solver.indices[edge])
            for identity in identities:
                owner = identity[0]
                for victim in victims:
                    if victim == owner:
                        continue
                    pairs.add((identity, victim, "track"))
                    owner_nets.add(owner)
                    victim_nets.add(victim)
                    involved_portals.add(portal_keys[identity])
                    history_nodes.update((source, target))

        id_to_name = {
            numeric_id: name
            for name, numeric_id in self.net_id_map.items()
        }
        for node, identities in node_owners.items():
            victim_ids = self._node_owner_members.get(node, ())
            if not victim_ids:
                continue
            for identity in identities:
                owner = identity[0]
                for victim_id in victim_ids:
                    victim = id_to_name.get(victim_id)
                    if victim is None or victim == owner:
                        continue
                    pairs.add((identity, victim, "via"))
                    owner_nets.add(owner)
                    victim_nets.add(victim)
                    involved_portals.add(portal_keys[identity])
                    history_nodes.add(node)

        return (
            pairs,
            owner_nets,
            victim_nets,
            involved_portals,
            history_nodes,
        )

    def _detect_path_node_conflicts(self):
        """Find different nets touching the same routed lattice node.

        Edge capacity is sufficient while each layer has only one planar
        axis. Guided routing adds the other axis, so perpendicular tracks can
        otherwise cross at a node without sharing an edge. The soft
        path-node cost normally prevents that state; this audit makes zero
        shared nodes an explicit convergence requirement.
        """
        if not hasattr(self, "path_node_use"):
            return set(), set(), {}

        shared_mask = self.path_node_use > 1
        shared_nodes = set(map(int, np.flatnonzero(shared_mask)))
        if not shared_nodes:
            return set(), set(), {}

        members = defaultdict(set)
        scores = defaultdict(int)
        for net_name, path in self.net_paths.items():
            if not path:
                continue
            graph_path = self._path_without_dynamic_escape_chains(
                net_name, path
            )
            nodes = self._unique_path_nodes(graph_path)
            if not nodes.size:
                continue
            hits = nodes[shared_mask[nodes]]
            for node in hits:
                members[int(node)].add(net_name)
                scores[net_name] += 1

        pairs = set()
        from itertools import combinations
        for node_members in members.values():
            pairs.update(combinations(sorted(node_members), 2))

        # path_node_use can only exceed one when at least two committed paths
        # contain the node. Keep the measured set aligned with the actual
        # paths if state was refreshed during a diagnostic call.
        measured_nodes = set(members)
        return pairs, measured_nodes, dict(scores)

    def _compute_path_node_overuse(self) -> Tuple[int, int]:
        """Return excess uses and over-capacity capacity-one graph nodes."""
        if not hasattr(self, "path_node_use"):
            return 0, 0
        overuse = np.maximum(
            np.asarray(self.path_node_use, dtype=np.int64) - 1,
            0,
        )
        return int(overuse.sum()), int(np.count_nonzero(overuse))

    def _path_node_layer_metrics(self) -> List[Dict[str, int]]:
        """Summarize capacity-one path-node occupancy by copper layer."""
        if not hasattr(self, "path_node_use") or not hasattr(
            self, "lattice"
        ):
            return []
        plane_size = int(
            self.lattice.x_steps * self.lattice.y_steps
        )
        layer_count = int(self.lattice.layers)
        use = np.asarray(self.path_node_use, dtype=np.int64)
        if (
            plane_size <= 0
            or layer_count <= 0
            or use.size != plane_size * layer_count
        ):
            return []

        metrics = []
        for layer in range(layer_count):
            layer_use = use[
                layer * plane_size:(layer + 1) * plane_size
            ]
            excess = np.maximum(layer_use - 1, 0)
            metrics.append({
                "layer": layer,
                "capacity_nodes": plane_size,
                "occupied_nodes": int(np.count_nonzero(layer_use)),
                "conflict_nodes": int(np.count_nonzero(excess)),
                "excess_uses": int(excess.sum()),
                "max_use": (
                    int(layer_use.max()) if layer_use.size else 0
                ),
            })
        return metrics

    @staticmethod
    def _negotiated_route_score(
        failed_nets: int,
        edge_via_overuse: int,
        path_node_overuse: int,
        physical_conflicts: int,
    ) -> Tuple[int, int, int]:
        """Rank routes by the complete negotiated resource system."""
        return (
            int(failed_nets),
            int(edge_via_overuse) + int(path_node_overuse),
            int(physical_conflicts),
        )

    def _detect_barrel_conflicts(self) -> Tuple[np.ndarray, int]:
        """
        Detect via barrel conflicts across all committed paths (GPU-accelerated).

        This is THE critical fix for shorting_items violations!
        Detects when committed edges touch via barrel nodes owned by other nets.

        Returns:
            (conflict_edge_indices, conflict_count)
        """
        import numpy as np

        logger.debug("[BARREL-CONFLICT] Checking for via barrel conflicts...")
        self._barrel_conflict_nets = set()
        self._barrel_owner_nets = set()
        self._barrel_victim_nets = set()
        self._barrel_owner_portal_keys = set()
        self._last_exact_barrel_conflict_count = 0
        self._last_path_node_conflict_count = 0
        self._last_escape_conflict_count = 0
        self._last_portal_grid_conflict_count = 0
        self._portal_grid_owner_nets = set()
        self._portal_grid_victim_nets = set()
        self._portal_grid_pairs = set()
        self._escape_conflict_pairs = set()
        self._exact_barrel_pairs = set()
        self._path_node_conflict_pairs = set()
        self._path_node_conflict_scores = {}
        self._physical_conflict_scores = defaultdict(int)
        self._last_exact_barrel_details = []

        # Bail out if node_owner not initialized
        if not hasattr(self, 'node_owner') or self.node_owner is None:
            logger.info("[BARREL-CONFLICT] Skipping: node_owner not initialized")
            return np.array([], dtype=np.int32), 0

        # Ensure edge_src mapping exists
        self._ensure_edge_src_map()

        # Use _net_paths (with underscore) - this is what the negotiation loop uses!
        paths_dict = getattr(self, '_net_paths', {})
        if not paths_dict:
            # Fallback to net_paths if _net_paths doesn't exist
            paths_dict = getattr(self, 'net_paths', {})

        if not paths_dict:
            logger.info("[BARREL-CONFLICT] Skipping: no committed paths found")
            return np.array([], dtype=np.int32), 0

        logger.info(f"[BARREL-CONFLICT] Found {len(paths_dict)} committed paths")

        # Collect cached committed edges in vectorized chunks. Recomputing
        # every path and extending Python lists one integer at a time made
        # this audit dominate large-board iterations.
        edge_chunks = []
        net_id_chunks = []

        for net_name, path in paths_dict.items():
            if not path or len(path) < 2:
                continue

            net_id = self._get_net_id(net_name)
            resource_edges = self._net_to_edges.get(net_name)
            if resource_edges is None:
                graph_path = self._path_without_dynamic_escape_chains(
                    net_name, path
                )
                # Exact physical-conflict measurement needs each traversal
                # once; _net_to_edges contains both directional CSR arcs.
                net_edges = self._path_to_directed_edges(graph_path)
                edge_chunk = np.asarray(net_edges, dtype=np.int32)
            else:
                edge_chunk = np.asarray(
                    resource_edges, dtype=np.int32
                )[:len(resource_edges) // 2]
            if edge_chunk.size == 0:
                continue
            edge_chunks.append(edge_chunk)
            net_id_chunks.append(
                np.full(edge_chunk.size, net_id, dtype=np.int32)
            )

        if not edge_chunks:
            logger.info("[BARREL-CONFLICT] No edges found in paths")
            return np.array([], dtype=np.int32), 0

        edge_indices = np.concatenate(edge_chunks)
        edge_net_ids = np.concatenate(net_id_chunks)
        logger.info(f"[BARREL-CONFLICT] Checking {len(edge_indices)} edges across {len(paths_dict)} nets")

        # SimpleDijkstra already owns the CPU CSR arrays. Reuse them instead
        # of downloading the full destination array from the GPU per audit.
        graph_indices_cpu = self.solver.indices

        # VECTORIZED CONFLICT DETECTION (GPU-accelerated!)
        # Get source and destination nodes for all edges at once
        src_nodes = self._edge_src[edge_indices]  # Vectorized lookup
        dst_nodes = graph_indices_cpu[edge_indices]  # Vectorized lookup

        # Get ownership for all nodes at once
        src_owners = self.node_owner[src_nodes]  # Vectorized lookup
        dst_owners = self.node_owner[dst_nodes]  # Vectorized lookup

        # Vectorized conflict check:
        # Conflict if src owned by different net OR dst owned by different net
        src_conflict = (src_owners != -1) & (src_owners != edge_net_ids)
        dst_conflict = (dst_owners != -1) & (dst_owners != edge_net_ids)
        conflict_mask = src_conflict | dst_conflict  # Element-wise OR

        logger.info(f"[BARREL-CONFLICT] Vectorized check of {len(edge_indices)} edges completed")

        # Get the actual edge indices that have conflicts
        conflict_edge_indices = edge_indices[conflict_mask]
        conflict_count = len(conflict_edge_indices)
        self._last_exact_barrel_conflict_count = conflict_count

        if conflict_count > 0:
            conflict_positions = np.flatnonzero(conflict_mask)
            victim_net_ids = set(
                map(int, edge_net_ids[conflict_positions])
            )
            owner_net_ids = set()
            conflict_node_chunks = []
            if np.any(src_conflict):
                conflict_node_chunks.append(
                    src_nodes[np.flatnonzero(src_conflict)]
                )
            if np.any(dst_conflict):
                conflict_node_chunks.append(
                    dst_nodes[np.flatnonzero(dst_conflict)]
                )
            conflict_nodes = np.unique(np.concatenate(
                conflict_node_chunks
            ))
            self._accumulate_node_conflict_history(conflict_nodes)
            for node_idx in conflict_nodes:
                owner_net_ids.update(
                    self._node_owner_members.get(int(node_idx), ())
                )
            id_to_name = {
                numeric_id: name
                for name, numeric_id in self.net_id_map.items()
            }
            exact_pairs = set()
            for position in conflict_positions:
                victim_id = int(edge_net_ids[position])
                conflicting_nodes = []
                if src_conflict[position]:
                    conflicting_nodes.append(int(src_nodes[position]))
                if dst_conflict[position]:
                    conflicting_nodes.append(int(dst_nodes[position]))
                for node_idx in conflicting_nodes:
                    for owner_id in self._node_owner_members.get(
                        node_idx, ()
                    ):
                        if owner_id == victim_id:
                            continue
                        victim_name = id_to_name.get(victim_id)
                        owner_name = id_to_name.get(owner_id)
                        if (
                            victim_name is not None
                            and owner_name is not None
                        ):
                            self._physical_conflict_scores[
                                victim_name
                            ] += 1
                            self._physical_conflict_scores[
                                owner_name
                            ] += 1
                            exact_pairs.add(tuple(sorted((
                                victim_name, owner_name
                            ))))
            self._exact_barrel_pairs = exact_pairs
            for position in conflict_positions[:20]:
                src_node = int(src_nodes[position])
                dst_node = int(dst_nodes[position])
                self._last_exact_barrel_details.append({
                    "victim": id_to_name.get(
                        int(edge_net_ids[position]),
                        str(int(edge_net_ids[position])),
                    ),
                    "edge": int(edge_indices[position]),
                    "src": src_node,
                    "dst": dst_node,
                    "src_owners": tuple(sorted(
                        id_to_name.get(owner, str(owner))
                        for owner in self._node_owner_members.get(
                            src_node, ()
                        )
                    )),
                    "dst_owners": tuple(sorted(
                        id_to_name.get(owner, str(owner))
                        for owner in self._node_owner_members.get(
                            dst_node, ()
                        )
                    )),
                })
            self._barrel_victim_nets = {
                id_to_name[numeric_id]
                for numeric_id in victim_net_ids
                if numeric_id in id_to_name
            }
            self._barrel_owner_nets = {
                id_to_name[numeric_id]
                for numeric_id in owner_net_ids
                if numeric_id in id_to_name
            }
            conflict_xy = {
                self.lattice.idx_to_coord(int(node_idx))[:2]
                for node_idx in conflict_nodes
            }
            for owner_net in self._barrel_owner_nets:
                selected = self.net_selected_portals.get(owner_net, ())
                pad_ids = self.net_pad_ids.get(owner_net, ())
                layers = self.net_portal_layers.get(owner_net, ())
                for pad_id, portal, entry_layer in zip(
                    pad_ids, selected, layers
                ):
                    if (
                        portal is not None
                        and (portal.x_idx, portal.y_idx) in conflict_xy
                    ):
                        self._barrel_owner_portal_keys.add((
                            pad_id,
                            portal.x_idx,
                            portal.y_idx,
                            int(entry_layer),
                        ))
            self._barrel_conflict_nets = {
                id_to_name[numeric_id]
                for numeric_id in (victim_net_ids | owner_net_ids)
                if numeric_id in id_to_name
            }
            logger.info(f"[BARREL-CONFLICT] Detected {conflict_count} conflicts (checked {len(edge_indices)} edges)")
        else:
            logger.info(f"[BARREL-CONFLICT] No conflicts found (checked {len(edge_indices)} edges)")

        (
            path_node_pairs,
            shared_path_nodes,
            path_node_scores,
        ) = self._detect_path_node_conflicts()
        self._path_node_conflict_pairs = set(path_node_pairs)
        self._path_node_conflict_scores = dict(path_node_scores)
        self._last_path_node_conflict_count = len(shared_path_nodes)
        if shared_path_nodes:
            for net_name, score in path_node_scores.items():
                self._physical_conflict_scores[net_name] += int(score)
            involved_nets = {
                net_name
                for pair in path_node_pairs
                for net_name in pair
            }
            self._barrel_conflict_nets.update(involved_nets)
            self._accumulate_node_conflict_history(shared_path_nodes)
            logger.info(
                "[PATH-NODE-CONFLICT] Detected %d shared nodes "
                "across %d net pairs",
                len(shared_path_nodes),
                len(path_node_pairs),
            )

        # Escape stubs and portal via bodies are off-lattice physical
        # geometry, so edge accounting cannot see their conflicts. Audit the
        # selected candidates with the same dimensions used for emission and
        # feed offenders into the normal negotiated hotset.
        self._rebuild_escape_occupancy()
        escape_pairs, escape_owners, escape_victims = (
            self._detect_escape_conflicts()
        )
        self._last_escape_conflict_count = len(escape_pairs)
        self._escape_conflict_pairs = set(escape_pairs)
        if escape_pairs:
            for first, second in escape_pairs:
                self._physical_conflict_scores[first[0]] += 1
                self._physical_conflict_scores[second[0]] += 1
            # Escape stubs live outside edge accounting, so their selected
            # candidates need their own Pathfinder history. Penalize both
            # ends of every physical conflict; otherwise these conflicts can
            # reroute forever without making either alternative more costly.
            self._barrel_owner_portal_keys.update(
                self._escape_conflict_portal_keys(escape_pairs)
            )
            self._barrel_owner_nets.update(escape_owners)
            self._barrel_victim_nets.update(escape_victims)
            self._barrel_conflict_nets.update(
                escape_owners | escape_victims
            )
            logger.info(
                "[ESCAPE-CONFLICT] Detected %d selected escape conflicts",
                len(escape_pairs),
            )

        (
            portal_grid_pairs,
            portal_grid_owners,
            portal_grid_victims,
            portal_grid_keys,
            portal_grid_nodes,
        ) = self._detect_portal_grid_conflicts()
        self._last_portal_grid_conflict_count = len(
            portal_grid_pairs
        )
        self._portal_grid_pairs = set(portal_grid_pairs)
        self._portal_grid_owner_nets = set(portal_grid_owners)
        self._portal_grid_victim_nets = set(portal_grid_victims)
        if portal_grid_pairs:
            for identity, victim, _kind in portal_grid_pairs:
                self._physical_conflict_scores[identity[0]] += 1
                self._physical_conflict_scores[victim] += 1
            self._barrel_owner_portal_keys.update(portal_grid_keys)
            self._barrel_owner_nets.update(portal_grid_owners)
            self._barrel_victim_nets.update(portal_grid_victims)
            self._barrel_conflict_nets.update(
                portal_grid_owners | portal_grid_victims
            )
            self._accumulate_node_conflict_history(
                portal_grid_nodes
            )
            logger.info(
                "[PORTAL-GRID-CONFLICT] Detected %d terminal-via "
                "conflicts (%d owners, %d victims)",
                len(portal_grid_pairs),
                len(portal_grid_owners),
                len(portal_grid_victims),
            )

        return (
            conflict_edge_indices,
            conflict_count
            + len(shared_path_nodes)
            + len(escape_pairs)
            + len(portal_grid_pairs),
        )

    def _path_is_manhattan(self, path: List[int]) -> bool:
        """Validate that path obeys Manhattan routing discipline"""
        for a, b in zip(path, path[1:]):
            x0, y0, z0 = self.lattice.idx_to_coord(a)
            x1, y1, z1 = self.lattice.idx_to_coord(b)
            if z0 == z1:
                # Planar move: must be adjacent (Manhattan distance = 1)
                if (abs(x1 - x0) + abs(y1 - y0)) != 1:
                    logger.error(f"[PATH-INVALID-DETAIL] Planar non-adjacent: ({x0},{y0},{z0}) -> ({x1},{y1},{z1}), dist={abs(x1-x0)+abs(y1-y0)}")
                    return False
            else:
                # Via move: same X,Y, any Z distance (allow multi-layer vias for portals)
                if not ((x1 == x0) and (y1 == y0)):
                    logger.error(f"[PATH-INVALID-DETAIL] Via with X/Y change: ({x0},{y0},{z0}) -> ({x1},{y1},{z1})")
                    return False
        return True

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

        # QUANTIZE: Round to grid to prevent float drift
        pitch = self.lattice.geom.pitch
        origin_x = self.lattice.geom.grid_min_x
        origin_y = self.lattice.geom.grid_min_y

        ax_mm = origin_x + round((ax_mm - origin_x) / pitch) * pitch
        ay_mm = origin_y + round((ay_mm - origin_y) / pitch) * pitch
        bx_mm = origin_x + round((bx_mm - origin_x) / pitch) * pitch
        by_mm = origin_y + round((by_mm - origin_y) / pitch) * pitch

        return {
            'net': net,
            'layer': self.config.layer_names[layer] if layer < len(self.config.layer_names) else f"L{layer}",
            'x1': ax_mm, 'y1': ay_mm, 'x2': bx_mm, 'y2': by_mm,
            'width': self.config.track_width,
        }

    def precompute_all_pad_escapes(self, board: Board, nets_to_route: List = None) -> Tuple[List, List]:
        """
        Delegate to PadEscapePlanner for precomputing all pad escapes.

        NEW: Pre-registers portal locations as via keepouts for iteration 1.
        This prevents tracks from routing through portal columns before escape vias are created.

        Returns (tracks, vias) for visualization.
        """
        if not self.escape_planner:
            logger.error("Escape planner not initialized! Call initialize_graph first.")
            return ([], [])

        result = self.escape_planner.precompute_all_pad_escapes(board, nets_to_route)

        # CRITICAL: Copy portals from escape_planner to self.portals
        # The pathfinder needs these to route from portal positions, not pad positions
        self.portals = self.escape_planner.portals.copy()
        self.portal_candidates = {
            pad_id: list(candidates)
            for pad_id, candidates
            in self.escape_planner.portal_candidates.items()
        }
        logger.info(f"Copied {len(self.portals)} portals from escape planner to pathfinder")

        # Cache escape geometry for merging into final payload
        self._escape_tracks, self._escape_vias = result
        logger.info(f"Cached {len(self._escape_tracks)} escape tracks and {len(self._escape_vias)} escape vias")

        # Track escape vias in via spatial arrays to prevent routing collisions
        self._track_escape_vias_in_via_usage()

        # NOTE: Portal via keepout pre-registration removed - too slow for full-graph
        # Via barrel conflicts exist but owner-aware blocking doesn't scale
        # TODO: Investigate actual root cause of dangling vias

        return result

    def _via_world(self, at_idx: int, net: str, from_layer: int, to_layer: int):
        x, y, _ = self.lattice.idx_to_coord(at_idx)
        (x_mm, y_mm) = self.lattice.geom.lattice_to_world(x, y)

        # CRITICAL FIX: Quantize via coordinates to grid (same as _segment_world)
        # This ensures via centers EXACTLY match track endpoints (no epsilon mismatch!)
        pitch = self.lattice.geom.pitch
        origin_x = self.lattice.geom.grid_min_x
        origin_y = self.lattice.geom.grid_min_y
        x_mm = origin_x + round((x_mm - origin_x) / pitch) * pitch
        y_mm = origin_y + round((y_mm - origin_y) / pitch) * pitch

        # Normalize layer order (consistent output, KiCad accepts either way)
        if from_layer > to_layer:
            from_layer, to_layer = to_layer, from_layer

        via = {
            'net': net,
            'x': x_mm, 'y': y_mm,
            'from_layer': self.config.layer_names[from_layer] if from_layer < len(self.config.layer_names) else f"L{from_layer}",
            'to_layer': self.config.layer_names[to_layer] if to_layer < len(self.config.layer_names) else f"L{to_layer}",
            'diameter': self.config.via_diameter,
            'drill': self.config.via_drill,
        }
        hdi_stack = getattr(self.config, "hdi_stack", None)
        if hdi_stack is not None:
            process = hdi_stack.process_for_span(
                from_layer, to_layer
            )
            via.update({
                "diameter": process.diameter_mm,
                "drill": process.drill_mm,
                "via_process": process.name,
                "via_kind": process.kind,
                "hdi_stack": hdi_stack.name,
            })
        return via

    def _expand_hdi_vias(self, vias: List[dict]) -> List[dict]:
        """Express every emitted HDI transition as legal physical spans."""
        hdi_stack = getattr(self.config, "hdi_stack", None)
        if hdi_stack is None:
            return list(vias)

        expanded = []
        for via in vias:
            from_layer = self._layer_name_to_index(
                via.get("from_layer")
            )
            to_layer = self._layer_name_to_index(
                via.get("to_layer")
            )
            if from_layer is None or to_layer is None:
                raise ValueError(
                    "HDI via has an unknown layer span: "
                    f"{via.get('from_layer')} -> {via.get('to_layer')}"
                )
            for physical_from, physical_to in hdi_stack.expand_span(
                from_layer, to_layer
            ):
                lo, hi = canonical_pair(physical_from, physical_to)
                process = hdi_stack.process_for_span(lo, hi)
                item = dict(via)
                item.update({
                    "from_layer": self.config.layer_names[lo],
                    "to_layer": self.config.layer_names[hi],
                    "diameter": process.diameter_mm,
                    "drill": process.drill_mm,
                    "via_process": process.name,
                    "via_kind": process.kind,
                    "hdi_stack": hdi_stack.name,
                })
                expanded.append(item)
        return expanded

    def _refresh_selected_escape_geometry(self) -> None:
        """Emit stubs only for the portal candidates selected by routing."""
        if self.escape_planner is None:
            return

        tracks = []
        vias = []
        emitted_pads = set()
        for net_id, pad_ids in self.net_pad_ids.items():
            if not self.net_paths.get(net_id):
                continue
            selected = self.net_selected_portals.get(net_id)
            if selected is None:
                selected = tuple(
                    self.portals.get(pad_id) for pad_id in pad_ids
                )
            layers = self.net_portal_layers.get(net_id, (1, 1))

            for pad_id, portal, entry_layer in zip(
                pad_ids, selected, layers
            ):
                if portal is None or pad_id in emitted_pads:
                    continue
                emitted_pads.add(pad_id)
                geometry = (
                    self.escape_planner._emit_portal_escape_geometry(
                        net_id,
                        pad_id,
                        portal,
                        entry_layer,
                        include_via=True,
                    )
                )
                for item in geometry:
                    if "x1" in item and "y1" in item:
                        tracks.append(item)
                    elif "x" in item and "y" in item:
                        vias.append(item)

        self._escape_tracks = tracks
        self._escape_vias = vias

    def emit_geometry(self, board: Board) -> Tuple[int, int]:
        """
        Convert routed node paths into drawable segments and vias.
        - Clean geometry (for KiCad export): only if overuse == 0
        - Provisional geometry (for GUI feedback): always generated

        CRITICAL: Escape geometry is ALWAYS merged, even with overuse.
        Escapes are the connection from pads to the routing grid and must be exported.
        """
        self._refresh_selected_escape_geometry()

        # Generate provisional geometry from routing paths
        provisional_tracks, provisional_vias = self._generate_geometry_from_paths()

        # ALWAYS merge escape geometry with routed geometry
        # Deduplicate helper
        def _dedupe(items, key_fn):
            seen, out = set(), []
            for it in items:
                k = key_fn(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        final_tracks = provisional_tracks
        final_vias = provisional_vias

        if hasattr(self, '_escape_tracks') and self._escape_tracks:
            # Merge escapes first (so they're visually "underneath")
            combined_tracks = self._escape_tracks + provisional_tracks
            combined_vias = self._escape_vias + provisional_vias

            # Deduplicate by geometric signature
            final_tracks = _dedupe(
                combined_tracks,
                lambda t: (t["net"], t["layer"],
                          round(t["x1"], 3), round(t["y1"], 3),
                          round(t["x2"], 3), round(t["y2"], 3),
                          round(t["width"], 3))
            )
            final_vias = _dedupe(
                combined_vias,
                lambda v: (v["net"], round(v["x"], 3), round(v["y"], 3),
                          v.get("from_layer"), v.get("to_layer"),
                          round(v.get("drill", 0), 3),
                          round(v.get("diameter", 0), 3))
            )

            logger.info(f"[ESCAPE-MERGE] escapes={len(self._escape_tracks)} + "
                       f"routed={len(provisional_tracks)} → "
                       f"total={len(final_tracks)} tracks after dedup")
            logger.info(f"[ESCAPE-MERGE] escape_vias={len(self._escape_vias)} + "
                       f"routed_vias={len(provisional_vias)} → "
                       f"total={len(final_vias)} vias after dedup")

        final_vias = self._expand_hdi_vias(final_vias)
        if getattr(self.config, "hdi_stack", None) is not None:
            final_vias = _dedupe(
                final_vias,
                lambda v: (
                    v["net"],
                    round(v["x"], 3),
                    round(v["y"], 3),
                    v.get("from_layer"),
                    v.get("to_layer"),
                    round(v.get("drill", 0), 4),
                    round(v.get("diameter", 0), 4),
                    v.get("via_process"),
                ),
            )

        # Store merged geometry as provisional (for GUI display)
        self._provisional_geometry = GeometryPayload(final_tracks, final_vias)

        # Check for overuse (include via spatial violations)
        over_sum, over_cnt = self.accounting.compute_overuse(router_instance=self)

        if over_sum > 0:
            logger.warning(f"[EMIT] Overuse={over_sum}: showing merged geometry in GUI but not exporting to KiCad")
            self._geometry_payload = GeometryPayload([], [])  # No clean geometry for export
            # Return merged counts so GUI shows escapes + routes
            return (len(final_tracks), len(final_vias))

        # No overuse: emit clean geometry for KiCad export
        logger.info("[EMIT] Routing converged! Exporting clean geometry with escapes")
        self._geometry_payload = GeometryPayload(final_tracks, final_vias)
        return (len(final_tracks), len(final_vias))

    def _path_without_dynamic_escape_chains(
        self, net_id: str, path: List[int]
    ) -> List[int]:
        """Remove terminal barrels supplied by explicit escape geometry."""
        selected = self.net_selected_portals.get(net_id)
        layers = self.net_portal_layers.get(net_id)
        if not path or selected is None or layers is None:
            return list(path)

        start = 0
        end = len(path)
        src_portal, dst_portal = selected
        src_layer, dst_layer = layers
        target = (
            src_portal.x_idx,
            src_portal.y_idx,
            src_layer,
        )
        for index, node in enumerate(path):
            if self.lattice.idx_to_coord(node) == target:
                start = index
                break
        target = (
            dst_portal.x_idx,
            dst_portal.y_idx,
            dst_layer,
        )
        for index in range(len(path) - 1, start - 1, -1):
            if self.lattice.idx_to_coord(path[index]) == target:
                end = index + 1
                break
        return list(path[start:end])

    def _generate_geometry_from_paths(self) -> Tuple[List, List]:
        """Generate tracks and vias from net_paths"""
        tracks, vias = [], []

        for net_id, path in self.net_paths.items():
            if not path:
                continue
            path = self._path_without_dynamic_escape_chains(
                net_id, path
            )
            if not path:
                continue
            if getattr(self.config, "hdi_stack", None) is None:
                path = self._coalesce_vertical_runs(path)

            # NOTE: Escape geometry is pre-computed by PadEscapePlanner and cached.
            # It will be merged with routed geometry in emit_geometry().

            # Generate tracks/vias from main path
            run_start = path[0]
            prev = path[0]
            prev_dir = None
            prev_layer = self.lattice.idx_to_coord(prev)[2]

            for node in path[1:]:
                x0, y0, z0 = self.lattice.idx_to_coord(prev)
                x1, y1, z1 = self.lattice.idx_to_coord(node)

                # Drop any planar segment on outer layers (shouldn't happen once graph/ROI are fixed)
                if z0 == z1 and (z0 == 0 or z0 == self.lattice.layers - 1):
                    logger.error(f"[EMIT-GUARD] refusing planar segment on outer layer {z0} for net {net_id}")
                    prev = node
                    prev_layer = z1
                    run_start = node
                    continue

                # VALIDATION: Check if nodes are adjacent (Manhattan distance should be 1)
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                dz = abs(z1 - z0)

                if dz == 0:  # Same layer - enforce H/V discipline
                    # Must be adjacent
                    if (dx + dy) != 1:
                        logger.error(f"[GEOMETRY-BUG] Non-adjacent nodes in path for net {net_id}: "
                                   f"({x0},{y0},{z0}) → ({x1},{y1},{z1}), Manhattan dist = {dx+dy}")
                        logger.error(f"[GEOMETRY-BUG] Path indices: prev={prev}, node={node}")
                        logger.error(f"[GEOMETRY-BUG] This creates diagonal segment! GPU parent pointers are CORRUPT!")
                        continue  # Skip illegal segment

                    # Check layer direction discipline
                    layer_axis = "h" if dy == 0 else "v"
                    if layer_axis not in self.lattice.get_allowed_axes(z0):
                        logger.error(
                            "[LAYER-VIOLATION] Axis %s is unavailable "
                            "on layer %d",
                            layer_axis,
                            z0,
                        )
                        continue
                    if layer_axis == 'h':
                        # H layer: y must be constant (horizontal movement)
                        if dy != 0:
                            logger.error(f"[LAYER-VIOLATION] H-layer {z0} has vertical move: "
                                       f"({x0},{y0})→({x1},{y1}), dy={dy}")
                            continue
                    else:  # 'v'
                        # V layer: x must be constant (vertical movement)
                        if dx != 0:
                            logger.error(f"[LAYER-VIOLATION] V-layer {z0} has horizontal move: "
                                       f"({x0},{y0})→({x1},{y1}), dx={dx}")
                            continue

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

        # FINAL VALIDATION: Check all tracks are axis-aligned
        violations = []
        for i, track in enumerate(tracks):
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            # Must be axis-aligned (one coordinate must be constant)
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 0.001 and dy > 0.001:
                violations.append((i, track, dx, dy))

        if violations:
            logger.error(f"[EMIT-VALIDATION] Found {len(violations)} diagonal segments!")
            for i, track, dx, dy in violations[:5]:  # Show first 5
                logger.error(f"  Track {i}: ({track['x1']:.2f},{track['y1']:.2f})->({track['x2']:.2f},{track['y2']:.2f}), "
                           f"Delta=({dx:.2f},{dy:.2f}) on {track['layer']}")

            # In debug mode, raise error
            if __debug__:
                raise RuntimeError(f"{len(violations)} diagonal segments detected at emission")
        else:
            logger.info(f"[EMIT-VALIDATION] All {len(tracks)} tracks are axis-aligned ✓")

        # Count tracks by layer and direction
        layer_stats = {}
        for track in tracks:
            layer = track['layer']
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            is_horizontal = (abs(y1 - y2) < 0.001)
            is_vertical = (abs(x1 - x2) < 0.001)

            if layer not in layer_stats:
                layer_stats[layer] = {'h': 0, 'v': 0}

            if is_horizontal:
                layer_stats[layer]['h'] += 1
            elif is_vertical:
                layer_stats[layer]['v'] += 1

        # Log per-layer statistics and check direction discipline
        for layer in sorted(layer_stats.keys()):
            h_count = layer_stats[layer]['h']
            v_count = layer_stats[layer]['v']
            logger.info(f"[LAYER-STATS] {layer}: {h_count} horizontal, {v_count} vertical")

            try:
                layer_index = self.config.layer_names.index(layer)
                expected_dir = self.lattice.get_legal_axis(layer_index)
                if expected_dir == 'h' and v_count > h_count:
                    logger.warning(
                        f"[LAYER-DIRECTION] {layer} is H-preferred "
                        "but has more V traces"
                    )
                elif expected_dir == 'v' and h_count > v_count:
                    logger.warning(
                        f"[LAYER-DIRECTION] {layer} is V-preferred "
                        "but has more H traces"
                    )
            except (ValueError, IndexError):
                pass

        return (tracks, vias)

    def _coalesce_vertical_runs(self, path: List[int]) -> List[int]:
        """Collapse adjacent z hops at one x/y into one physical via span."""
        if len(path) < 3:
            return list(path)

        result = [path[0]]
        via_xy = None
        via_direction = 0

        for previous, node in zip(path, path[1:]):
            x0, y0, z0 = self.lattice.idx_to_coord(previous)
            x1, y1, z1 = self.lattice.idx_to_coord(node)
            dz = z1 - z0
            is_vertical = x0 == x1 and y0 == y1 and dz != 0
            direction = int(np.sign(dz)) if is_vertical else 0

            if (
                is_vertical
                and via_xy == (x0, y0)
                and direction == via_direction
            ):
                result[-1] = node
            else:
                result.append(node)

            if is_vertical:
                via_xy = (x0, y0)
                via_direction = direction
            else:
                via_xy = None
                via_direction = 0

        return result

    def get_geometry_payload(self):
        """
        Get geometry payload for GUI/export.

        Returns clean geometry if available (no overuse),
        otherwise returns provisional geometry so GUI can still display/export.
        """
        # If clean geometry is empty but provisional exists, return provisional
        if (not self._geometry_payload.tracks and not self._geometry_payload.vias
            and hasattr(self, '_provisional_geometry')
            and (self._provisional_geometry.tracks or self._provisional_geometry.vias)):
            return self._provisional_geometry
        return self._geometry_payload

    def get_provisional_geometry(self):
        """Get provisional geometry for GUI feedback (always available)"""
        return self._provisional_geometry


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY API
# ═══════════════════════════════════════════════════════════════════════════════

UnifiedPathFinder = PathFinderRouter

logger.info(f"PathFinder loaded (GPU={'YES' if GPU_AVAILABLE else 'NO'})")
