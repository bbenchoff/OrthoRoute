# PathFinder Negotiation Re-implementation Checklist

**LOST WORK:** All PathFinder negotiation improvements were in working directory and lost during git reset. Need to re-implement from documented specifications.

## 1. CENTRALIZED CONFIGURATION (Remove Environment Variables)

### Constants to Add at Top of File:
```python
# ============================================================================
# PATHFINDER CONFIGURATION - ALL PARAMETERS IN ONE PLACE
# ============================================================================

# Grid and Geometry Parameters
GRID_PITCH = 0.4                    # Grid pitch in mm for routing lattice
LAYER_COUNT = 6                     # Number of copper layers

# PathFinder Algorithm Parameters
BATCH_SIZE = 32                     # Number of nets processed per batch
MAX_ITERATIONS = 30                 # Maximum PathFinder negotiation iterations
MAX_SEARCH_NODES = 50000            # Maximum nodes explored per net
PER_NET_BUDGET_S = 0.5             # Time budget per net in seconds
MAX_ROI_NODES = 20000              # Maximum nodes in Region of Interest

# PathFinder Cost Parameters
PRES_FAC_INIT = 1.0                # Initial present factor for congestion
PRES_FAC_MULT = 1.6                # Present factor multiplier per iteration
PRES_FAC_MAX = 1000.0              # Maximum present factor cap
HIST_ACCUM_GAIN = 1.0              # Historical cost accumulation gain
OVERUSE_EPS = 1e-6                 # Epsilon for overuse calculations

# Algorithm Tuning Parameters
DELTA_MULTIPLIER = 4.0             # Delta-stepping bucket size multiplier
ADAPTIVE_DELTA = True              # Enable adaptive delta tuning
STRICT_CAPACITY = True             # Enforce strict capacity constraints
REROUTE_ONLY_OFFENDERS = True      # Reroute only offending nets in incremental mode

# Via and Routing Parameters
VIA_COST = 0.0                     # Cost penalty for vias (0 = no penalty)
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

# Fixed Seed for Deterministic Routing
ROUTING_SEED = 42                  # Fixed seed for reproducible results

# Negotiation Parameters
STAGNATION_PATIENCE = 5            # Iterations without improvement before stopping
STRICT_OVERUSE_BLOCK = False       # Block overused edges with infinite cost (CHANGED FROM TRUE)
HIST_COST_WEIGHT = 1.0             # Weight for historical cost component
OVERUSE_WEIGHT = 6.0               # Explicit overuse penalty weight

# Debugging and Profiling Parameters
ENABLE_PROFILING = False           # Enable performance profiling
ENABLE_INSTRUMENTATION = False     # Enable detailed instrumentation
```

### PathFinderConfig Updates:
```python
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
    overuse_eps: float = OVERUSE_EPS
    mode: str = "delta_stepping"
    roi_parallel: bool = False
    per_net_budget_s: float = PER_NET_BUDGET_S
    max_roi_nodes: int = MAX_ROI_NODES
    delta_multiplier: float = DELTA_MULTIPLIER
    grid_pitch: float = GRID_PITCH
    adaptive_delta: bool = ADAPTIVE_DELTA
    strict_capacity: bool = STRICT_CAPACITY
    reroute_only_offenders: bool = REROUTE_ONLY_OFFENDERS
    layer_count: int = LAYER_COUNT
    enable_profiling: bool = ENABLE_PROFILING
    enable_instrumentation: bool = ENABLE_INSTRUMENTATION
    stagnation_patience: int = STAGNATION_PATIENCE
    strict_overuse_block: bool = STRICT_OVERUSE_BLOCK
    hist_cost_weight: float = HIST_COST_WEIGHT
    overuse_weight: float = OVERUSE_WEIGHT
    # Diagnostics toggles (used in your loop)
    log_iteration_details: bool = False
    # Cost weights (you use this in _update_edge_total_costs)
    acc_fac: float = 0.0
    # Phase control parameters
    phase_block_after: int = 4   # was 2 — give soft mode more time
    congestion_multiplier: float = 1.0
```

## 2. PROPER 4-PHASE NEGOTIATION LOOP

### Replace _pathfinder_negotiation method with:
```python
def _pathfinder_negotiation(self, valid_nets: Dict[str, Tuple[int, int]], progress_cb=None, total=0) -> Dict[str, List[int]]:
    """PathFinder negotiation loop with proper 4-phase iteration: refresh → cost update → route → commit"""
    cfg = self.config
    pres_fac = cfg.pres_fac_init
    best_unrouted = None
    stagnant_iters = 0

    # Clear per-iter present, but keep STORE (history of the current round)
    self._reset_present_usage()               # present = 0
    # Note: DO NOT clear store usage - it persists between iterations

    # Mark that negotiation is running
    self._negotiation_ran = True
    logger.info(f"[NEGOTIATE] start: iters={cfg.max_iterations} pres={pres_fac:.2f}×{cfg.pres_fac_mult:.2f}")

    self.routed_nets.clear()
    total_nets = len(valid_nets)

    for it in range(1, cfg.max_iterations + 1):
        logger.info("[NEGOTIATE] iter=%d pres_fac=%.2f", it, pres_fac)
        self.current_iteration = it

        # Track path changes for stagnation detection
        import numpy as np
        old_paths = {
            net_id: (np.asarray(path, dtype=np.int64).copy()
                     if path is not None else np.empty(0, dtype=np.int64))
            for net_id, path in self._net_paths.items()
        }

        # 1) Pull last iteration's result into PRESENT
        mapped = self._refresh_present_usage_from_store()    # logs how many entries mapped
        self._check_overuse_invariant("iter-start", compare_to_store=True)

        # Sanity check after refresh
        logger.info("[SANITY] iter=%d store_edges=%d present_nonzero=%d",
                    self.current_iteration,
                    len(self._edge_store),
                    int((self.edge_present_usage > 0).sum()))

        # 2) Compute overuse on PRESENT and update costs
        over_sum, over_edges = self._compute_overuse_stats_present()  # must not raise
        self._update_edge_total_costs(pres_fac)

        # 3) Route all nets against current costs (must not throw on single-net failure)
        routed_ct, failed_ct = self._route_all_nets_cpu_in_batches_with_metrics(valid_nets, progress_cb)

        # Calculate how many nets changed paths this iteration
        routes_changed = 0
        for net_id in valid_nets:
            old_path = old_paths.get(net_id, np.empty(0, dtype=np.int64))
            new_path = self._net_paths.get(net_id, [])
            if not np.array_equal(np.asarray(old_path), np.asarray(new_path)):
                routes_changed += 1

        logger.info("[ROUTES-CHANGED] %d nets changed this iter", routes_changed)

        # 4) Commit PRESENT → STORE so next iter sees it
        changed = self._commit_present_usage_to_store()

        # Sanity check after commit
        logger.info("[SANITY] iter=%d store_edges=%d present_nonzero=%d",
                    self.current_iteration,
                    len(self._edge_store),
                    int((self.edge_present_usage > 0).sum()))

        logger.info("[ITER-RESULT] routed=%d failed=%d overuse_edges=%d over_sum=%d changed=%s",
                    routed_ct, failed_ct, over_edges, over_sum, bool(changed))

        # ---- Termination logic ----
        # Success: no overuse and no failures
        if failed_ct == 0 and over_edges == 0:
            logger.info("[NEGOTIATE] Converged: all nets routed with legal usage.")
            self._routing_result = {'success': True, 'needs_more_layers': False}
            return dict(self.routed_nets)

        # Track "no progress" to avoid spinning forever
        cur_unrouted = failed_ct + (1 if over_edges > 0 else 0)
        if best_unrouted is None or cur_unrouted < best_unrouted:
            best_unrouted = cur_unrouted
            stagnant_iters = 0
        else:
            stagnant_iters += 1

        # Optional early stop on stagnation
        if stagnant_iters >= cfg.stagnation_patience:
            logger.warning("[NEGOTIATE] Stagnated for %d iters (best_unrouted=%d).",
                           stagnant_iters, best_unrouted)
            break

        # Increase present-cost pressure and loop
        pres_fac *= cfg.pres_fac_mult

    # Fell out of loop: decide the message
    self._routing_result = self._finalize_insufficient_layers()  # compute & store analysis
    return dict(self.routed_nets)
```

## 3. RIP-UP/REROUTE LOGIC

### Add to __init__:
```python
# Track each net's current path (CSR edge indices)
self._net_paths = {}   # net_id -> np.ndarray of edge indices (CSR), dtype=int64
```

### Helper Methods:
```python
def _prepare_net_for_reroute(self, net_id):
    prev = self._net_paths.get(net_id)
    if prev is None or len(prev) == 0:
        return None
    import numpy as np
    idx = np.asarray(prev, dtype=np.int64)
    np.subtract.at(self.edge_present_usage, idx, 1)
    # if you maintain owners, free them for this net:
    if hasattr(self, "edge_owners"):
        for e in idx.tolist():
            if self.edge_owners.get(e) == net_id:
                del self.edge_owners[e]
    return idx  # return so we can restore on failure

def _restore_net_after_failed_reroute(self, net_id, prev_idx):
    if prev_idx is None:
        return
    import numpy as np
    np.add.at(self.edge_present_usage, prev_idx, 1)
    if hasattr(self, "edge_owners"):
        for e in prev_idx.tolist():
            self.edge_owners[e] = net_id
```

## 4. STORE/PRESENT PERSISTENCE

### Exact Methods:
```python
def _reset_present_usage(self):
    """Zero the per-iteration usage vector (length = E_live)."""
    if hasattr(self, 'edge_present_usage') and self.edge_present_usage is not None:
        self.edge_present_usage.fill(0)

def _refresh_present_usage_from_store(self) -> int:
    """Overwrite PRESENT from canonical store. No merging. No side channels."""
    E = self.edge_present_usage.shape[0]
    self.edge_present_usage.fill(0)

    store = getattr(self, '_edge_store', None)
    if not store:
        return 0

    # store is dict: {edge_idx: count}
    # guard for bad indices
    import numpy as np
    idxs = np.fromiter(store.keys(), dtype=np.int64, count=len(store))
    vals = np.fromiter((int(v) for v in store.values()), dtype=np.int32, count=len(store))
    good = (idxs >= 0) & (idxs < E)
    if good.any():
        self.edge_present_usage[idxs[good]] = vals[good]
        return int(vals[good].sum())
    return 0

def _commit_present_usage_to_store(self) -> bool:
    """Canonicalize PRESENT → STORE (replace)."""
    import numpy as np
    nz = np.nonzero(self.edge_present_usage)[0]
    vals = self.edge_present_usage[nz].astype(int, copy=False)

    store = self._edge_store  # dict[int]->int
    # detect change
    before = len(store)

    store.clear()
    if nz.size:
        # ~10x faster than per-item in Python loops, but still fine on 10–100k
        for i, v in zip(nz.tolist(), vals.tolist()):
            if v:
                store[i] = int(v)

    return len(store) != before
```

## 5. FIXED LAYER SHORTFALL ESTIMATOR

### Replace _estimate_layer_shortfall:
```python
def _estimate_layer_shortfall(self) -> int:
    """
    Return a conservative, local estimate of extra layers needed.
    Interpretation: one new layer adds one extra track per edge (per direction),
    so the layer shortfall equals the worst overuse at any single edge.
    """
    import numpy as np
    usage = np.asarray(self.edge_present_usage, dtype=np.int32)
    cap   = np.asarray(self.edge_capacity,       dtype=np.int32)
    if usage.size == 0 or cap.size == 0:
        return 0

    over = usage - cap
    # clamp negatives and handle empty safely
    if hasattr(np, "maximum"):
        over = np.maximum(over, 0)
    else:
        over[over < 0] = 0

    # CRITICAL FIX: MAX not SUM
    max_over = int(over.max(initial=0)) if hasattr(over, "max") else int(over.max())

    # Keep it non-negative; 0 means "no extra layers required"
    # Cap it to reasonable bound (don't scream "126 layers" at users)
    shortfall = max(0, max_over)
    return int(np.clip(shortfall, 0, 16))
```

## 6. COST CALCULATION FIX

### Update _update_edge_total_costs:
```python
def _update_edge_total_costs(self, pres_fac: float) -> None:
    """Update total edge costs for PathFinder negotiation using present cost factor."""
    import numpy as np
    usage = self.edge_present_usage
    cap   = self.edge_capacity
    hist  = self.edge_history
    base  = self.edge_base_cost
    legal = getattr(self, "edge_dir_mask", None)
    # normalize to a NumPy bool array (CPU or GPU)
    if legal is None:
        legal = np.ones_like(base, dtype=bool)
    else:
        if hasattr(legal, "get"):  # CuPy → NumPy
            legal = legal.get()
        legal = legal.astype(bool, copy=False)

    # Ensure numpy, not device arrays
    if hasattr(usage, "get"): usage = usage.get()
    if hasattr(cap,   "get"): cap   = cap.get()
    if hasattr(hist,  "get"): hist  = hist.get()
    if hasattr(base,  "get"): base  = base.get()

    over = np.maximum(usage - cap, 0.0)

    # NEW: explicit overuse weight (configurable)
    total = base \
          + pres_fac * usage \
          + self.config.overuse_weight * over \
          + self.config.hist_cost_weight * hist

    # Hard-block illegal edges
    total[~legal] = np.inf

    # Strict DRC: also block explicit overuse immediately (only in HARD phase)
    if self.current_iteration >= self.config.phase_block_after and self.config.strict_overuse_block:
        over_mask = usage > cap
        total[over_mask] = np.inf

    self.edge_total_cost = total
```

## 7. FRIENDLY ERROR MESSAGES

### Add helper method:
```python
def _format_failure_message(self) -> str:
    r = getattr(self, "_routing_result", None)
    # Works for either dataclass or dict
    get = (lambda k, default=None:
            getattr(r, k, default) if hasattr(r, k) else
            (r.get(k, default) if isinstance(r, dict) else default))

    if r and (get("needs_more_layers", False) or get("success", False) is False):
        unrouted     = get("unrouted", 0)
        over_edges   = get("overuse_edges", 0)
        over_sum     = get("overuse_sum", 0)
        shortfall    = get("layer_shortfall", 0)
        return (f"[INSUFFICIENT-LAYERS] Unrouted={unrouted}, "
                f"overuse_edges={over_edges}, over_sum={over_sum}. "
                f"Estimated additional layers needed: {shortfall}. "
                "Increase layer count or relax design rules.")
    return "[ROUTING-FAIL] No routes produced; check constraints and design rules."
```

## 8. CRITICAL FIXES

### Fix edge_mask reference:
- Change `self.edge_mask.astype(bool)` to proper edge_dir_mask handling

### Fix R-tree imports:
- Change `index.Index()` to `rtree_index.Index()`

### Fix variable naming:
- Change `hist_cost` to `edge_history` throughout

### Fix capacity array usage:
- Change scalar capacity checks to array-based: `cap[idx]` not `self._edge_capacity`

### Remove double-counting:
- Remove duplicate PRESENT += 1 writes in batch routing

### Add phase tracking:
- Initialize `self.current_iteration = 0` in __init__

## 9. VERIFICATION TESTS

### Import test:
```python
python -c "import orthoroute.algorithms.manhattan.unified_pathfinder as upf; print('IMPORT_OK', hasattr(upf, 'UnifiedPathFinder'))"
```

### Grid pitch test:
```python
config = upf.PathFinderConfig()
print(f'Grid pitch: {config.grid_pitch}')  # Should be 0.4
```

### Negotiation test:
```python
pf = upf.UnifiedPathFinder(config=config, use_gpu=False)
# Should have all the new methods and not crash
```

## IMPLEMENTATION ORDER:
1. Centralized configuration first (fixes crashes)
2. Cost calculation and phase gating
3. Store/present persistence methods
4. Negotiation loop replacement
5. Rip-up/reroute logic
6. Layer shortfall fix
7. Error message improvements
8. Final testing and verification

**TARGET:** Router that negotiates properly, shows realistic layer estimates (1-16 not 126), and gives friendly error messages.