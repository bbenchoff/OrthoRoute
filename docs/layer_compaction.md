# Layer Compaction for PathFinder Routing

**Status:** Research / Proposed Feature
**Last Updated:** November 14, 2025
**Related:** Post-routing optimization, cost reduction

---

## Problem Statement

After routing a complex backplane with PathFinder on 32 layers, analysis may reveal:

```
Layer Utilization Analysis:
  F.Cu    (0):  ████████████████████░░  85% utilized
  In1.Cu  (1):  ███████████████░░░░░░░  65% utilized
  In2.Cu  (2):  ████████████████░░░░░░  70% utilized
  In3.Cu  (3):  ███████████░░░░░░░░░░░  55% utilized
  ...
  In14.Cu (14): ██████░░░░░░░░░░░░░░░░  30% utilized
  In15.Cu (15): ████░░░░░░░░░░░░░░░░░░  20% utilized
  In16.Cu (16): ███░░░░░░░░░░░░░░░░░░░  15% utilized
  In17.Cu (17): ██░░░░░░░░░░░░░░░░░░░░  10% utilized
  In18.Cu (18): █░░░░░░░░░░░░░░░░░░░░░   5% utilized
  In19-30:      ░░░░░░░░░░░░░░░░░░░░░░  <5% utilized
  B.Cu    (31): ████████████████████░░  80% utilized
```

**Question:** Can we "pull up" traces from lower layers (17-30) to upper layers (1-16) that have unused capacity, reducing the total layer count from 32 to perhaps 20?

**Why this matters:**
- **Cost:** PCB fab pricing scales with layer count (~$50 more per 2 layers)
- **Stackup:** Simpler stackups are easier to manufacture and have better yields
- **Lead time:** Fewer layers = faster fabrication
- **Example:** Routing with 32 layers but only using 20 could save $300 per board

---

## What Is Layer Compaction?

**Layer Compaction** (also called **Layer Minimization** or **Track Layer Reassignment**) is a post-routing optimization that attempts to reduce the number of physical layers required while maintaining routing completeness and DRC compliance.

### Related Problems in VLSI/PCB CAD

**1. Layer Assignment Problem (VLSI)**
- During global routing, routes are found in 2D without layer assignment
- Layer assignment step maps 2D routes to specific 3D metal layers
- Goal: Minimize layers while respecting routing density constraints
- **NP-hard** in general case but solvable with heuristics

**2. Track Assignment Problem**
- Intermediate step between global and detailed routing
- Assigns routes to specific tracks on each layer
- Can optimize for layer utilization, crosstalk, timing

**3. Via Migration**
- Post-routing optimization that changes via layer spans
- Example: Via from F.Cu→In20.Cu becomes F.Cu→In12.Cu
- Requires rerouting affected trace segments

---

## Why PathFinder Doesn't Automatically Minimize Layers

PathFinder's **negotiated congestion** algorithm optimizes for:
1. ✅ **Routability** - All nets successfully routed
2. ✅ **Wirelength** - Minimize total trace length
3. ✅ **Congestion** - Eliminate overuse/conflicts
4. ✅ **Via count** - Reduce layer transitions

But it does **NOT** optimize for:
- ❌ **Layer count minimization** - Uses all available layers equally
- ❌ **Layer utilization balance** - Doesn't prefer upper layers
- ❌ **Unused layer elimination** - Doesn't avoid sparse layers

### Why?

**PathFinder's cost function:**
```
cost(edge) = base_cost + pres_fac * present_usage + hist_fac * history
```

This makes all layers equally attractive (same base cost). Adding:
```
layer_penalty = (layer_depth / max_layers)^2
```

Would bias routing toward upper layers, but PathFinder would use lower layers when upper layers get congested.

**Result:** PathFinder distributes routing load across all layers for maximum routability, not minimum layer count.

---

## Academic Background

### Research on Layer Assignment

From literature review:

**"Layer assignment is a desirable intermediate step between global routing and detailed routing"** (Track Assignment research)

**Approaches:**
1. **ILP-based** - Integer Linear Programming for optimal assignment (slow)
2. **Greedy heuristics** - Fast but suboptimal
3. **Simulated annealing** - Good quality, moderate runtime
4. **Graph-based** - Model as min-cost flow or multicommodity flow

**BoxRouter 2.0 approach:**
- Perform 2D global routing first (planar)
- Layer assignment as second phase
- Machine learning to predict optimal layer ordering

**Key insight:** Separating 2D routing from layer assignment allows explicit layer minimization.

---

## Proposed Approaches

### Approach 1: PathFinder-Based Compaction (RECOMMENDED)

**Key insight:** Use PathFinder itself to reroute nets with layer constraints!

**Workflow:**
```bash
# New mode in main.py:
python main.py compact TestBackplane.ORS --target-layers 20
```

**What it does:**
1. Load ORS file → reconstruct routing solution
2. Initialize PathFinder with board geometry (from ORP file)
3. Load existing routing into PathFinder internal state
4. Set layer constraints (forbid layers > 20)
5. Identify nets using forbidden layers (layers 21-31)
6. Rip up those nets
7. Reroute with PathFinder (respecting layer constraints)
8. Export new ORS file with compacted solution

**Advantages:**
- ✅ **Full PathFinder power** - Real negotiated congestion routing
- ✅ **Can resolve conflicts** - Negotiation handles overlaps
- ✅ **High success rate** - PathFinder will find paths if they exist
- ✅ **Quality routing** - Optimal paths within constraints
- ✅ **Existing infrastructure** - Reuses ORS import/export

**Implementation:**
```python
def run_compact(ors_file, target_layers, max_iterations=50):
    # Load ORS and corresponding ORP
    geometry_data, metadata = import_solution_from_ors(ors_file)
    orp_file = ors_file.replace('.ORS', '.ORP')
    board_data = import_board_from_orp(orp_file)

    # Initialize PathFinder
    board = create_board_from_data(board_data)
    pf = UnifiedPathFinder(config=PathFinderConfig(max_iterations=max_iterations))
    pf.initialize_graph(board)
    pf.prepare_routing_runtime()

    # Load existing solution into PathFinder
    load_solution_into_pathfinder(pf, geometry_data)

    # Set layer constraints
    pf.set_max_layer_constraint(target_layers)

    # Find and reroute deep nets
    deep_nets = find_nets_using_layers_above(geometry_data, target_layers)
    for net_id in deep_nets:
        pf.rip_up_net(net_id)
    pf.route_multiple_nets(deep_nets)

    # Export compacted solution
    geom = pf.get_geometry_payload()
    output_file = ors_file.replace('.ORS', f'_compact_{target_layers}L.ORS')
    export_solution_to_ors(geom, metadata, output_file)
```

See detailed implementation in **Approach 1 Implementation** section below.

---

### Approach 2: ORS-Only Post-Processor (Simpler, Limited)

**Alternative:** Geometry manipulation without rerouting.

### High-Level Strategy

```
Input: Fully routed board with N layers
Output: Same routing using M < N layers (M minimal)

1. Analyze layer utilization
2. Identify migration candidates (traces on underutilized layers)
3. Attempt to migrate traces to upper layers with free capacity
4. Verify DRC compliance
5. Iterate until no more migrations possible
6. Report final layer count
```

### Algorithm Design

#### Phase 1: Layer Utilization Analysis

```python
def analyze_layer_utilization(pathfinder):
    """
    Compute routing density per layer.

    Returns:
        layer_stats = {
            0: {'tracks': 250, 'capacity': 1000, 'util': 0.25},
            1: {'tracks': 180, 'capacity': 1000, 'util': 0.18},
            ...
        }
    """
    layer_stats = {}

    for layer_id in range(pathfinder.lattice.Nz):
        # Count tracks on this layer
        tracks_on_layer = [t for t in geometry.tracks if t['layer'] == layer_id]

        # Estimate routing capacity (function of board area and grid pitch)
        capacity = estimate_layer_capacity(pathfinder.lattice, layer_id)

        layer_stats[layer_id] = {
            'tracks': len(tracks_on_layer),
            'capacity': capacity,
            'utilization': len(tracks_on_layer) / capacity,
            'free_capacity': capacity - len(tracks_on_layer)
        }

    return layer_stats
```

#### Phase 2: Identify Target Layer Count

```python
def find_minimal_layer_count(layer_stats, min_util_threshold=0.05):
    """
    Find deepest layer with >= 5% utilization.

    Example: If layers 0-19 have >5% util, but 20-31 have <5%,
             target is 20 layers.
    """
    max_used_layer = 0
    for layer_id, stats in sorted(layer_stats.items(), reverse=True):
        if stats['utilization'] >= min_util_threshold:
            max_used_layer = layer_id
            break

    # Add safety margin (need some headroom for migration)
    target_layer_count = max_used_layer + 3

    return target_layer_count
```

#### Phase 3: Migration Strategy

**Greedy approach (fast, good for first pass):**

```python
def compact_to_target_layers(pathfinder, target_layer_count):
    """
    Migrate traces from deep layers to upper layers.

    Strategy: Start with deepest layers, migrate upward.
    """
    migrations_attempted = 0
    migrations_succeeded = 0

    # Process layers from deepest to shallowest (above target)
    for source_layer in range(pathfinder.lattice.Nz - 1, target_layer_count - 1, -1):
        # Get all net segments on this layer
        segments = find_segments_on_layer(pathfinder, source_layer)

        for net_id, segment in segments:
            # Find best target layer (respecting H/V constraints)
            target_layer = find_migration_target(
                segment,
                max_layer=target_layer_count - 1,
                layer_stats=layer_stats,
                direction=get_segment_direction(segment)
            )

            if target_layer is None:
                continue

            migrations_attempted += 1

            # Attempt migration
            success = migrate_segment(
                pathfinder,
                net_id,
                segment,
                source_layer,
                target_layer
            )

            if success:
                migrations_succeeded += 1
                update_layer_stats(layer_stats, segment, source_layer, target_layer)

    return migrations_attempted, migrations_succeeded
```

---

## Key Challenges

### 1. Manhattan H/V Layer Constraints

**Problem:** PathFinder uses alternating H/V layers:
- Odd layers (1, 3, 5...): Horizontal routing only
- Even layers (2, 4, 6...): Vertical routing only

**Implication:** Can't migrate a horizontal segment from In17.Cu (odd) to In16.Cu (even).

**Solution:**
```python
def find_migration_target(segment, max_layer, direction):
    """
    Find target layer respecting H/V constraints.

    If segment is horizontal:
        - Can only move to odd layers (1, 3, 5...)
    If segment is vertical:
        - Can only move to even layers (2, 4, 6...)
    """
    candidates = []
    for layer in range(1, max_layer):
        layer_direction = 'H' if layer % 2 == 1 else 'V'
        if layer_direction == direction:
            candidates.append(layer)

    # Sort by available capacity
    return select_best_layer(candidates, layer_stats)
```

### 2. Via Barrel Constraints

**Problem:** Moving a trace from In20.Cu to In10.Cu changes via barrels.

**Example:**
```
Before migration:
  Via A: F.Cu (0) → In20.Cu (20)   [drills through layers 0-20]
  Trace: In20.Cu horizontal segment
  Via B: In20.Cu (20) → B.Cu (31)  [drills through layers 20-31]

After migration to In10.Cu:
  Via A: F.Cu (0) → In10.Cu (10)   [drills through layers 0-10] ✓ Shorter!
  Trace: In10.Cu horizontal segment
  Via B: In10.Cu (10) → B.Cu (31)  [drills through layers 10-31] ✓ Shorter!
```

**Benefits:**
- ✅ Shorter via barrels = less capacitance
- ✅ Fewer layers drilled = less manufacturing cost
- ✅ Better signal integrity (less via stub)

**Gotcha - Barrel Conflicts:**

If another net has a via at the same (x, y) position:
```
Net 1: F.Cu → In10.Cu  [barrel: 0-10]
Net 2: In8.Cu → In15.Cu [barrel: 8-15]
         ^^^^^^^ OVERLAP at layers 8-10!
```

This creates a **barrel conflict** - two vias physically overlapping.

**Solution:** Migration must check for barrel conflicts at new layer depth.

### 3. Segment Connectivity

**Problem:** Can't migrate a segment in isolation - must consider net topology.

**Example:**
```
Net "VCC" path:
  Pad A (F.Cu) → [via] → In20.Cu → [segment] → [via] → In5.Cu → ... → Pad B

Migrating the In20.Cu segment requires:
  1. Changing first via from (F.Cu → In20.Cu) to (F.Cu → In10.Cu)
  2. Rerouting segment on In10.Cu
  3. Changing second via from (In20.Cu → In5.Cu) to (In10.Cu → In5.Cu)
```

**Complexity:** Must migrate entire connected component, not just one segment.

### 4. Capacity Estimation

**Challenge:** How much "free capacity" does a layer have?

**Naive approach:**
```python
free_capacity = total_routing_tracks - used_tracks
```

**Problems:**
- Doesn't account for routing density (congested areas)
- Assumes uniform distribution (not realistic)
- Ignores DRC spacing requirements

**Better approach:**
```python
# Divide layer into grid cells (e.g. 5mm x 5mm)
for cell in grid_cells:
    local_tracks = count_tracks_in_cell(cell, layer)
    local_capacity = estimate_cell_capacity(cell, layer)
    cell_utilization = local_tracks / local_capacity

# Layer can accept migration only if:
#   - Average utilization < 70%
#   - No cells > 90% utilized
#   - Migration target area < 80% utilized
```

---

## Implementation Strategy

### Option 1: Layer-Biased Routing (During Routing)

**Modify PathFinder cost function:**

```python
def compute_edge_cost(edge, layer):
    base = edge.base_cost
    present = pres_fac * edge.present_usage
    history = hist_fac * edge.history

    # NEW: Add layer depth penalty
    layer_penalty = (layer / max_layers) ** 2  # Quadratic penalty

    return base + present + history + layer_penalty
```

**Effect:** PathFinder naturally prefers upper layers during routing.

**Pros:**
- ✅ Simple to implement
- ✅ Integrated into routing (no post-processing)
- ✅ Respects congestion automatically

**Cons:**
- ❌ May sacrifice routability (might need those deep layers)
- ❌ Can't control target layer count explicitly
- ❌ Might increase wirelength

### Option 2: Post-Routing Layer Compaction (After Routing)

**Two-pass approach:**

**Pass 1: Conservative Migration**
```python
def conservative_layer_compaction(pathfinder, target_layers):
    """
    Only migrate segments where there's obvious free capacity.
    No rip-up/reroute - just layer reassignment.
    """
    for layer in range(target_layers, pathfinder.lattice.Nz):
        segments = get_segments_on_layer(pathfinder, layer)

        for net_id, seg in segments:
            # Find target layer with same direction and free space
            target = find_free_layer(seg, max_layer=target_layers - 1)

            if target and has_space_for_segment(target, seg):
                # Direct reassignment (no rerouting needed)
                reassign_segment_layer(pathfinder, net_id, seg, target)
```

**Pass 2: Aggressive Compaction with Rerouting**
```python
def aggressive_layer_compaction(pathfinder, target_layers):
    """
    Rip up and reroute nets that still use deep layers.
    Use PathFinder with layer constraints.
    """
    # Identify nets still using layers >= target_layers
    deep_nets = find_nets_using_deep_layers(pathfinder, target_layers)

    # Set layer constraints (forbid layers >= target_layers)
    pathfinder.set_layer_constraints(max_layer=target_layers - 1)

    # Rip up deep nets
    for net_id in deep_nets:
        pathfinder.rip_up_net(net_id)

    # Reroute with layer constraint
    pathfinder.route_multiple_nets(deep_nets, max_iterations=20)

    # Verify all nets routed within target layers
    verify_layer_compliance(pathfinder, target_layers)
```

---

## Algorithm Pseudocode

### Complete Layer Compaction Flow

```python
def compact_layers(pathfinder, board, min_layer_util=0.05):
    """
    Post-routing layer compaction to minimize PCB layer count.

    Args:
        pathfinder: UnifiedPathFinder instance with completed routing
        board: Board object
        min_layer_util: Minimum utilization to consider layer "used"

    Returns:
        CompactionResult with before/after metrics
    """

    # ===== PHASE 1: ANALYSIS =====
    print("[COMPACT] Phase 1: Analyzing layer utilization...")

    layer_stats = analyze_layer_utilization(pathfinder)
    original_layers = pathfinder.lattice.Nz

    # Find minimal layer count needed
    target_layers = find_minimal_layer_count(layer_stats, min_layer_util)

    if target_layers >= original_layers - 2:
        print(f"[COMPACT] Board already optimal: using {original_layers} layers efficiently")
        return CompactionResult(success=False, reason="Already optimized")

    print(f"[COMPACT] Target: Reduce from {original_layers} to {target_layers} layers")
    print(f"[COMPACT] Potential savings: {(original_layers - target_layers) * 2} layers")

    # ===== PHASE 2: CONSERVATIVE MIGRATION =====
    print("[COMPACT] Phase 2: Conservative migration (no rerouting)...")

    mig_attempted, mig_succeeded = 0, 0

    for source_layer in range(target_layers, original_layers):
        segments = get_all_segments_on_layer(pathfinder, source_layer)
        print(f"[COMPACT] Layer {source_layer}: {len(segments)} segments to migrate")

        for net_id, segment in segments:
            # Find suitable target layer
            direction = 'H' if segment['x1'] != segment['x2'] else 'V'
            target = find_target_layer(
                direction,
                max_layer=target_layers - 1,
                layer_stats=layer_stats
            )

            if target is None:
                continue

            mig_attempted += 1

            # Check if migration is safe (no DRC violations)
            if can_migrate_safely(pathfinder, net_id, segment, target):
                migrate_segment_to_layer(pathfinder, net_id, segment, target)
                mig_succeeded += 1

    print(f"[COMPACT] Conservative migration: {mig_succeeded}/{mig_attempted} succeeded")

    # ===== PHASE 3: AGGRESSIVE REROUTING =====
    print("[COMPACT] Phase 3: Aggressive rerouting for remaining deep nets...")

    # Find nets still using layers >= target_layers
    deep_nets = []
    for net_id, path in pathfinder.net_paths.items():
        if uses_layers_above(path, target_layers):
            deep_nets.append(net_id)

    if not deep_nets:
        print("[COMPACT] All nets migrated successfully!")
        return finalize_compaction(pathfinder, target_layers)

    print(f"[COMPACT] {len(deep_nets)} nets still use deep layers - attempting reroute")

    # Set layer mask (forbid layers >= target_layers)
    original_mask = pathfinder.get_layer_mask()
    restricted_mask = create_layer_mask(max_layer=target_layers - 1)
    pathfinder.set_layer_mask(restricted_mask)

    # Rip up and reroute with layer constraint
    for net_id in deep_nets:
        pathfinder.rip_up_net(net_id)

    reroute_result = pathfinder.route_multiple_nets(
        deep_nets,
        max_iterations=30,
        strict_layer_compliance=True
    )

    # Restore original mask
    pathfinder.set_layer_mask(original_mask)

    # ===== PHASE 4: VERIFICATION =====
    print("[COMPACT] Phase 4: Verification...")

    failed_nets = verify_layer_compliance(pathfinder, target_layers)

    if failed_nets:
        print(f"[COMPACT] WARNING: {len(failed_nets)} nets could not be migrated")
        print(f"[COMPACT] Best achievable: {max_layer_used(pathfinder) + 1} layers")
        return CompactionResult(
            success=False,
            achieved_layers=max_layer_used(pathfinder) + 1,
            failed_nets=failed_nets
        )

    # Success!
    actual_layers = max_layer_used(pathfinder) + 1
    layers_saved = original_layers - actual_layers

    print(f"[COMPACT] ✓ SUCCESS! Reduced from {original_layers} to {actual_layers} layers")
    print(f"[COMPACT] Savings: {layers_saved} layers (~${layers_saved * 25} per board)")

    return CompactionResult(
        success=True,
        original_layers=original_layers,
        final_layers=actual_layers,
        layers_saved=layers_saved,
        migrations=mig_succeeded
    )
```

---

## Integration with PathFinder

### Where to Hook In

**Option A: Automatic post-routing**
```python
# In unified_pathfinder.py, after emit_geometry():
if config.enable_layer_compaction:
    compaction_result = compact_layers(self, board)
    if compaction_result.success:
        # Re-emit geometry with compacted layers
        self.emit_geometry(board)
```

**Option B: Manual trigger (GUI)**
```python
# In main_window.py, add menu item:
def optimize_layer_usage(self):
    """Layer compaction post-processing"""
    if not self.router:
        return

    # Show analysis dialog
    layer_stats = analyze_layer_utilization(self.router)

    msg = f"Current layers: {self.board_data['layer_count']}\n"
    msg += f"Estimated minimal: {find_minimal_layer_count(layer_stats)}\n"
    msg += f"\nProceed with layer compaction?"

    if QMessageBox.question(self, "Layer Compaction", msg) == QMessageBox.Yes:
        result = compact_layers(self.router, target_layers)
        # Update display
        self.pcb_viewer.update()
```

**Option C: Headless flag**
```bash
python main.py headless board.ORP --compact-layers 20
```

---

## Expected Results

### Test Case: 18-Layer Backplane

**Before compaction:**
```
Layer utilization:
  Layers 0-10:  60-80% utilized
  Layers 11-15: 20-40% utilized
  Layers 16-17: <10% utilized

Total: 18 layers
```

**After compaction:**
```
Layer utilization:
  Layers 0-10:  65-85% utilized (slight increase)
  Layers 11-15: 35-55% utilized (moderate increase)
  Layers 16-17: ELIMINATED

Total: 16 layers (saved 2 layers = ~$50 per board)
```

### Performance Impact

**Expected changes:**
- **Wirelength:** +2-5% (longer paths to avoid congestion)
- **Via count:** +5-10% (more layer transitions)
- **Routing time:** +10-20% (additional rerouting pass)
- **DRC violations:** 0 (must maintain zero violations)

**Trade-off analysis:**
- Small increases in wirelength/vias
- Significant cost savings for production

---

## Implementation Roadmap

### Phase 1: Analysis Tools (1-2 hours)

- [ ] `analyze_layer_utilization()` - Compute per-layer stats
- [ ] `find_minimal_layer_count()` - Determine target
- [ ] `generate_utilization_report()` - Text/CSV output
- [ ] Add to GUI: "Analyze Layer Usage" button

**Deliverable:** Users can see layer utilization without running compaction.

### Phase 2: Simple Migration (3-4 hours)

- [ ] `find_segments_on_layer()` - Extract segments by layer
- [ ] `find_migration_target()` - Find target layer respecting H/V
- [ ] `migrate_segment_to_layer()` - Reassign without rerouting
- [ ] `verify_layer_compliance()` - Check all segments within target

**Deliverable:** Conservative migration for obvious wins.

### Phase 3: PathFinder Integration (4-6 hours)

- [ ] `set_layer_mask()` - Constrain routing to specific layers
- [ ] `rip_up_net()` - Remove net path from graph
- [ ] Modify `_route_single_net()` to respect layer mask
- [ ] Add layer constraint to cost function

**Deliverable:** Can reroute nets with layer restrictions.

### Phase 4: Full Compaction Algorithm (6-8 hours)

- [ ] Implement complete `compact_layers()` function
- [ ] Add conflict resolution (barrel conflicts, DRC)
- [ ] Iterative refinement loop
- [ ] Performance metrics and reporting

**Deliverable:** Complete layer compaction feature.

### Phase 5: UI and Testing (2-3 hours)

- [ ] GUI menu item: "Optimize Layer Count..."
- [ ] Progress dialog during compaction
- [ ] Before/after comparison view
- [ ] Test on multiple board sizes

**Total estimated time: 16-23 hours**

---

## Open Questions

### 1. Blind/Buried Via Support

Current OrthoRoute uses **through-hole vias** only. Layer compaction could benefit from:
- **Blind vias:** F.Cu → In10.Cu (doesn't drill to B.Cu)
- **Buried vias:** In5.Cu → In15.Cu (doesn't reach outer layers)

**Trade-off:** Adds complexity and cost, but enables better layer utilization.

**Recommendation:** Phase 1 implementation assumes through-hole vias only. Add blind/buried support in Phase 2.

### 2. Layer Assignment During Routing

Alternative approach: Instead of post-processing, integrate layer bias into routing:

```python
# In PathFinder cost function:
layer_depth_penalty = lambda z: (z / max_z) ** 2 * depth_weight

# Prefer upper layers during routing
edge_cost += layer_depth_penalty(edge.layer) * 0.5
```

**Pros:** More natural, better integration
**Cons:** Might sacrifice routability

**Recommendation:** Try both approaches, compare results.

### 3. What if Compaction Fails?

If aggressive rerouting can't fit all nets in target layers:

**Options:**
1. **Accept partial compaction** - Reduce 32 → 28 instead of 32 → 20
2. **Relax constraints incrementally** - Try 24 layers, then 26, then 28...
3. **Report failure** - User decides whether to accept current routing

**Recommendation:** Implement progressive fallback (try 20, then 22, then 24...).

### 4. Signal Integrity Impact

Moving high-speed signals to different layers affects:
- **Impedance:** Different layer stackup positions have different impedance
- **Crosstalk:** Moving closer to other signals increases coupling
- **Return paths:** Ground/power plane proximity changes

**Recommendation:**
- Phase 1: Ignore SI (optimize for layer count only)
- Phase 2: Add SI constraints (don't migrate critical nets)

---

## Cost-Benefit Analysis

### PCB Fabrication Cost vs. Layer Count

**Typical pricing (6-layer baseline):**
```
 6 layers: $100 per board
 8 layers: $125 per board (+25%)
10 layers: $160 per board (+60%)
12 layers: $200 per board (+100%)
14 layers: $250 per board (+150%)
16 layers: $320 per board (+220%)
18 layers: $400 per board (+300%)
20 layers: $490 per board (+390%)
...
32 layers: $1200 per board (+1100%)
```

**Savings example:**
- Compacting 32 → 24 layers: **~$400 per board saved**
- Production run of 100 boards: **$40,000 total savings**

### When Layer Compaction Makes Sense

**Good candidates:**
- ✅ Boards routed with "safety margin" (32 layers available, only 20 needed)
- ✅ Prototype-to-production transition (optimize for cost)
- ✅ Low-speed backplanes (no critical SI requirements)
- ✅ Large production runs (optimization cost amortized)

**Poor candidates:**
- ❌ Already tightly routed (no free capacity)
- ❌ High-speed designs (SI constraints dominate)
- ❌ One-off prototypes (optimization time not worth it)
- ❌ Mixed signal boards (analog/power layers can't be compacted)

---

## Research References

Based on web search and VLSI literature:

1. **PathFinder Algorithm (McMurchie & Ebeling, 1995)**
   - Negotiated congestion routing for FPGAs
   - Iterative rip-up and reroute
   - OrthoRoute's foundation algorithm

2. **Layer Assignment in VLSI**
   - "Layer assignment is a desirable intermediate step between global routing and detailed routing"
   - Typically done as 2D routing + layer assignment (BoxRouter 2.0)
   - Machine learning approaches for optimal layer ordering

3. **Track Assignment Algorithms**
   - SPTA 2.0: Scalable parallel ILP-based track assignment
   - RDTA: Routability-driven track assignment
   - Can optimize for layer utilization

4. **PCB Industry Practice**
   - Layer minimization typically done during design phase
   - Manual optimization by expert layout engineers
   - Post-routing layer reduction not common in commercial tools

---

## Proposed Feature Specification

### User Interface

**Menu Item:** `Route → Optimize Layer Count...`

**Dialog:**
```
┌─────────────────────────────────────────────────┐
│  Layer Compaction Analysis                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  Current layer count: 32 layers                │
│  Estimated minimal: 22 layers                  │
│                                                 │
│  Layer utilization:                            │
│    Layers 0-15:   60-85% ████████████░░        │
│    Layers 16-21:  15-40% ███░░░░░░░░░░         │
│    Layers 22-31:  < 5%   █░░░░░░░░░░░░         │
│                                                 │
│  Estimated impact:                             │
│    Wirelength increase: ~3%                    │
│    Via count increase: ~50 vias                │
│    Layer savings: 10 layers                    │
│    Cost savings: ~$250 per board               │
│                                                 │
│  Target layer count: [22] (editable)           │
│                                                 │
│  [Cancel]  [Proceed with Compaction]           │
└─────────────────────────────────────────────────┘
```

**Progress Display:**
```
Compacting layers...
  Phase 1: Analyzing utilization... ✓
  Phase 2: Conservative migration... 150/200 segments migrated
  Phase 3: Rerouting deep nets... 15/20 nets rerouted
  Phase 4: Verification... ✓

Result: Successfully compacted to 22 layers!
```

### Command-Line Interface

```bash
# Analyze only (no modification)
python main.py headless board.ORP --analyze-layers

# Compact to target
python main.py headless board.ORP --compact-layers 20

# Compact to minimum
python main.py headless board.ORP --compact-layers auto
```

---

## Success Metrics

### How to Measure Success

**Primary metrics:**
- ✅ **Layer reduction:** Original layers → Final layers
- ✅ **Routability maintained:** 100% nets still routed
- ✅ **DRC compliance:** Zero violations

**Secondary metrics:**
- ⚠️ **Wirelength delta:** % increase (acceptable: <5%)
- ⚠️ **Via count delta:** # additional vias (acceptable: <10%)
- ⚠️ **Runtime:** Time spent on compaction (acceptable: <20% of routing time)

**Cost metrics:**
- 💰 **Fabrication cost saved:** $ per board
- 💰 **Break-even point:** # of boards to justify development effort

### Example Results

**Test board: 512-net backplane**
```
Before compaction:
  Layers: 18
  Tracks: 4088
  Vias: 2552
  Utilization: Layers 0-12 @ 70%, Layers 13-17 @ 15%

After compaction:
  Layers: 14 (saved 4 layers)
  Tracks: 4203 (+2.8% wirelength)
  Vias: 2687 (+5.3% vias)
  Utilization: Layers 0-13 @ 55-80%

Cost savings: $100 per board × 50 boards = $5,000
Runtime: +4.2 minutes (18% of routing time)
```

---

## Future Enhancements

### Advanced Optimizations

**1. Differential Pair Awareness**
- Keep diff pairs on same layer
- Migrate pairs together
- Maintain length matching

**2. Impedance Control**
- Preserve impedance-controlled layers
- Don't migrate high-speed signals to incompatible stackup positions

**3. Power Integrity**
- Don't migrate traces that need proximity to power/ground planes
- Preserve return path quality

**4. Thermal Considerations**
- Hot components might need specific layer routing
- Thermal vias can't be arbitrarily reassigned

**5. Manufacturing Constraints**
- Aspect ratio limits (drill depth / hole diameter)
- Via-in-pad restrictions
- Annular ring requirements

---

## Comparison to Other Approaches

### Layer Assignment During Global Routing (BoxRouter 2.0)

**Approach:** 2D routing first, then assign layers
**Pros:** Can optimize layer count explicitly
**Cons:** PathFinder is already 3D - would require major rewrite

### Via Minimization (Different Problem)

**Goal:** Reduce via count (not layer count)
**Method:** Prefer longer runs on same layer
**Relation:** Complementary to layer compaction

### Layer Swapping (FPGA-specific)

**Context:** FPGAs have limited routing tracks per layer
**Method:** Swap track assignments between layers
**Applicability:** Limited for PCB (different constraints)

---

## Conclusion

**Layer compaction is feasible for PathFinder-routed boards** and could provide significant cost savings for production runs.

**Recommended implementation priority:**
1. **High priority:** Analysis tools (show layer utilization)
2. **Medium priority:** Conservative migration (low-hanging fruit)
3. **Low priority:** Aggressive rerouting (complex, diminishing returns)

**Best use case:** Post-routing optimization for production cost reduction when board has been over-provisioned with layers during design.

**Next steps:**
1. Implement `analyze_layer_utilization()` function
2. Add "Layer Utilization Report" to GUI
3. Prototype conservative migration on test boards
4. Evaluate cost/benefit on real designs

---

**Status:** Ready for implementation
**Estimated development time:** 2-3 weeks for complete feature
**Expected ROI:** High for production boards with >50 unit runs

---

## Appendix: Mathematical Formulation

### Layer Assignment as Optimization Problem

**Variables:**
- `x[s,l]` = 1 if segment `s` assigned to layer `l`, 0 otherwise

**Objective:**
```
Minimize: max_layer_used
Subject to:
  - Each segment assigned to exactly one layer
  - Layer capacity not exceeded
  - H/V direction constraints satisfied
  - Via barrel conflicts = 0
  - Connectivity maintained (segment endpoints match via layers)
```

**Complexity:** NP-hard (reduction from graph coloring)

**Practical solution:** Greedy heuristics + local search

---

**Last updated:** November 14, 2025
**Author:** Research based on VLSI routing literature and PathFinder algorithm analysis
**Status:** Proposed feature for OrthoRoute v2.0
--- Approach 1: Detailed Implementation ---
