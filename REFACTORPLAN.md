# PathFinder Sequential Routing Refactor Plan

## 🎯 FOR THE NEXT LLM: START HERE

### Quick Context
**Problem discovered:** The GPU PathFinder implementation routes nets in **parallel batches with frozen costs**, breaking PathFinder's core negotiation mechanism. This causes:
- Vertical band gaps (all nets pick same "cheap" corridors)
- Routing stagnation (same nets succeed/fail every iteration)
- Large board failure (4.6% success vs expected 70%+)

**Root cause:** All nets in a batch route with the SAME cost snapshot. Net 512 cannot see pressure from nets 1-511, so they dogpile the same paths.

**Solution:** Implement **micro-batch negotiation** - small batches (8-16 nets) with cost updates between each batch.

---

### Testing Instructions

**Run small board test:**
```bash
python main.py --test-manhattan > test_YOURNAME.log 2>&1 &
```

**Monitor progress:**
```bash
# Check iterations (should take 1-3 minutes for baseline, 5-15 min for sequential)
grep "ITER [0-9].*routed=" test_YOURNAME.log

# Look for success message
grep -E "(TEST PASSED|All nets routed)" test_YOURNAME.log

# Check batch behavior
grep -E "(GPU-BATCH|MICRO-BATCH|SEQUENTIAL)" test_YOURNAME.log | head -20
```

**Success criteria:**
- Routed nets should **increase** across iterations (not stuck at same number)
- Final success rate should be 70%+ (vs baseline 50-55%)
- GUI popup should show "All nets routed with zero overuse" (screenshot validation)

**Look for this in logs:**
- `[CLEAN] All nets routed with zero overuse` (line 2385 of unified_pathfinder.py)
- Final iteration showing `overuse=0`

---

### Baseline Performance (BEFORE Changes)

**Test results from 2025-10-24:**

| Configuration | Method | Iter 1 | Iter 10 | Iter 30 | Time | Notes |
|--------------|--------|--------|---------|---------|------|-------|
| **Baseline (Stock)** | Large batches, frozen costs | 182/512 (35.5%) | 280/512 (54.7%) | 258/512 (50.4%) | ~3 min | Plateaus around 50%, can't improve |
| **Micro-batch (8 nets)** | Small batches + cost updates | 43/512 (8.4%) | ? | 59/512 (11.5%) | Slow | **WORSE** - costs too aggressive |
| **Sequential (batch=1)** | One net, no cost updates | 40/512 (7.8%) | 20/512 (3.9%) | 8/512 (1.6%) | Very slow | **TERRIBLE** - GPU overhead |
| **Sequential (batch=1) + updates** | One net + cost updates | CRASH | - | - | - | 8.6M log lines, stuck in iter 1 |

**Key finding:** Cost updates between batches made things WORSE (43 nets vs 182 baseline). The micro-batch approach needs careful tuning to avoid over-penalizing later nets.

---

## 📊 PROBLEM ANALYSIS

### What's Actually Happening

**Current flow (unified_pathfinder.py:2286-2400):**
```python
for iteration in 1..30:
    # STEP 1: Refresh accounting
    accounting.refresh_from_canonical()

    # STEP 2: ⚠️ COMPUTE COSTS ONCE (line 2327-2329)
    accounting.update_costs(base_costs, pres_fac, hist_weight)
    costs_FROZEN = accounting.total_cost.get()  # ← Snapshot for entire iteration!

    # STEP 3: Build hotset (nets touching overused edges)
    hotset = _build_hotset(tasks)  # ~400-500 nets

    # STEP 4: ⚠️ ROUTE ALL WITH FROZEN COSTS (line 2346 → 2738)
    routed, failed = _route_all(hotset, costs_FROZEN)
        → _route_all_batched_gpu(nets, costs_FROZEN)
            → Batch 1: Route nets 1-64 with costs_FROZEN
            → Batch 2: Route nets 65-128 with costs_FROZEN  ⚠️ Can't see batch 1!
            → ... (8 total batches)

    # STEP 5: Update history
    accounting.update_history(gain=1.0)

    # STEP 6: Escalate pressure
    pres_fac *= 2.0
```

**The smoking gun:**
- **Line 2729**: `costs = self.accounting.total_cost.get()` - Costs frozen here
- **Line 2738**: All batches receive same frozen `costs`
- **Line 3411**: GPU kernel uses frozen costs for all K nets in batch
- **No cost recomputation** happens until next iteration

### Classic PathFinder (Sequential)

What PathFinder is SUPPOSED to do (from McMurchie & Ebeling 1995):

```
for iteration:
    for net in nets (sequential order):
        cost(edge) = base + pres_fac * current_overuse(edge) + history(edge)
        path = route_with_costs(net, cost)
        commit(path)  # ← Increments present[edges]
        # Next net immediately sees updated present in cost formula
```

**Key insight:** Cost depends on **current present state**, which changes after every net routes. This creates pressure feedback that spreads nets across the fabric.

### Why Our GPU Implementation Fails

**Frozen costs prevent negotiation:**
1. Net 1 routes, uses edge E
2. Net 2 routes with SAME costs (doesn't see edge E is now used)
3. Net 2 also picks edge E (it's still "cheap")
4. Both nets commit to edge E → overuse
5. Overuse stays stuck because all nets make the same mistake

**Why large boards fail catastrophically:**
- 8192 nets routing in ~16 batches with frozen costs
- All nets in batch pick the "cheapest" path (same vertical corridor)
- No spread across iterations because costs don't reflect reality
- Same 375 nets succeed every iteration

---

## 🔧 ARCHITECTURAL CHANGES REQUIRED

### Option A: True Sequential (Pure PathFinder)

**Pros:**
- Exact PathFinder algorithm implementation
- Maximum negotiation quality
- Simple to understand

**Cons:**
- **10-20× slower** (512 GPU calls vs 8)
- GPU massively underutilized

**Implementation:**
```python
# unified_pathfinder.py:2706-2750 (_route_all)

def _route_all(self, tasks, all_tasks, pres_fac):
    ordered_nets = self._order_nets_by_difficulty(tasks)
    routed_total = 0

    for net_id in ordered_nets:
        src, dst = tasks[net_id]

        # ✅ COMPUTE COSTS WITH LIVE PRESENT STATE
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight
        )
        costs_live = self.accounting.total_cost.get()

        # Extract ROI
        roi_nodes, g2r = self.roi_extractor.extract_roi_geometric(src, dst)
        roi_csr = self._build_roi_csr(roi_nodes, costs_live)

        # Route SINGLE net with GPU (K=1)
        paths = self.solver.gpu_solver.find_paths_on_rois([roi_csr])

        # Commit (updates present immediately)
        if paths and paths[0]:
            edges = self._path_to_edges(paths[0])
            self.accounting.commit_path(edges)  # present[edges] += 1
            self.net_paths[net_id] = paths[0]
            routed_total += 1

    return routed_total, len(tasks) - routed_total
```

**Files modified:**
- `unified_pathfinder.py:2706-2750` (20 lines changed)
- No other files!

**Effort:** 3-4 hours

---

### Option B: Micro-Batch Hybrid (RECOMMENDED)

**Pros:**
- Balances negotiation quality and GPU efficiency
- 3-5× slower (vs 10-20× for sequential)
- Tunable via MICRO_BATCH_SIZE parameter

**Cons:**
- Still has intra-batch parallelism (weaker than pure sequential)
- More complex implementation

**Implementation:**
```python
# unified_pathfinder.py:2706-2750 (_route_all)

def _route_all(self, tasks, all_tasks, pres_fac):
    ordered_nets = self._order_nets_by_difficulty(tasks)
    total = len(ordered_nets)
    cfg = self.config

    MICRO_BATCH_SIZE = cfg.micro_batch_size  # 8, 16, or 32
    routed_total = 0
    failed_total = 0

    # ✅ MICRO-BATCH LOOP WITH COST UPDATES
    for batch_start in range(0, total, MICRO_BATCH_SIZE):
        batch_end = min(batch_start + MICRO_BATCH_SIZE, total)
        batch_nets = ordered_nets[batch_start:batch_end]

        # ✅ RECOMPUTE COSTS FOR THIS MICRO-BATCH
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs_for_batch = self.accounting.total_cost.get()

        # Route micro-batch with GPU (K=8-32)
        routed, failed = self._route_batch_gpu(
            batch_nets, tasks, all_tasks, costs_for_batch, pres_fac
        )

        routed_total += routed
        failed_total += failed

        # Costs will be recomputed for next batch (implicit)

    return routed_total, failed_total


# unified_pathfinder.py:2964 (rename function)
# BEFORE: def _route_all_batched_gpu(self, ordered_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus)
# AFTER:  def _route_batch_gpu(self, batch_nets, tasks, all_tasks, costs, pres_fac)

def _route_batch_gpu(self, batch_nets, tasks, all_tasks, costs, pres_fac):
    """Route a single micro-batch (K=8-32 nets) with given costs"""

    # Remove: outer batch loop (lines 3061-3600)
    # Keep: All ROI extraction, GPU dispatch, result processing
    # Change: Process single batch_nets list instead of looping

    cfg = self.config
    K = len(batch_nets)

    # Build ROI batch for K nets
    roi_batch = []
    for net_id in batch_nets:
        src, dst = tasks[net_id]
        # ... ROI extraction logic (keep as-is) ...
        roi_batch.append((src, dst, roi_indptr, roi_indices, costs, roi_size))

    # Single GPU call for K nets
    paths = self.solver.gpu_solver.find_paths_on_rois(roi_batch)

    # Process results
    routed = 0
    for net_id, path in zip(batch_nets, paths):
        if path:
            edges = self._path_to_edges(path)
            self.accounting.commit_path(edges)  # Updates present!
            self.net_paths[net_id] = path
            routed += 1

    return routed, K - routed
```

**Files modified:**
- `unified_pathfinder.py:2706-2750` - Add micro-batch loop (~30 lines)
- `unified_pathfinder.py:2964-3600` - Simplify to single-batch function (~50 lines removed)
- `config.py` - Add `micro_batch_size: int = 16` parameter (~1 line)

**Effort:** 6-8 hours coding + 4-6 hours testing = **1.5-2 days**

---

### Option C: CPU Fallback Test (Proof of Concept)

**Purpose:** Prove that sequential routing with cost updates actually works

**How:** The CPU fallback path (lines 2740-2900) already implements sequential routing with live costs!

**Quick test:**
```python
# unified_pathfinder.py:2732
# BEFORE:
use_gpu_batching = (hasattr(self.solver, 'gpu_solver') and ...)

# AFTER:
use_gpu_batching = False  # Force CPU sequential
```

This will be **slow** but prove the concept. If CPU fallback achieves 80%+ success, we know sequential negotiation is THE fix.

**Effort:** 30 seconds to test

---

## 🔍 DETAILED CODE ANALYSIS

### Key Files in manhattan/

**Core orchestration:**
- `unified_pathfinder.py` (2700 lines) - Main PathFinder logic
  - Lines 754-860: EdgeAccountant (accounting/cost tracking)
  - Lines 2286-2530: `_pathfinder_negotiation()` (iteration loop)
  - Lines 2706-2750: `_route_all()` - **WHERE COST FREEZING HAPPENS**
  - Lines 2964-3600: `_route_all_batched_gpu()` - Parallel batch routing

**GPU solver:**
- `cuda_dijkstra.py` (4500 lines) - GPU Dijkstra implementation
  - Lines 1879-1947: `find_paths_on_rois()` - Entry point
  - Lines 162-300: CUDA kernel code (embedded C++)
  - No changes needed for micro-batch!

**Support modules (no changes needed):**
- `pathfinder/roi_extractor_mixin.py` - ROI extraction
- `pathfinder/negotiation_mixin.py` - Hotset building
- `pathfinder/graph_builder_mixin.py` - CSR graph construction
- `pad_escape_planner.py` - Portal placement

### EdgeAccountant Deep Dive

**Location:** `unified_pathfinder.py:754-860`

**Data structures:**
```python
self.canonical: Dict[edge_idx → count]  # Ground truth (persistent)
self.present: Array[E]                  # Current iteration usage
self.history: Array[E]                  # Accumulated congestion
self.total_cost: Array[E]               # Routing costs (computed)
```

**Key methods:**
- `refresh_from_canonical()` (line 768): Rebuild present from canonical
- `commit_path(edges)` (line 775): Increments both canonical AND present
- `clear_path(edges)` (line 782): Decrements both
- `update_costs()` (line 838): **THE COST FORMULA**
  ```python
  over = max(0, present - capacity)
  total_cost = base * base_weight + pres_fac * over + hist_weight * history
  ```

**Critical insight:** `commit_path()` already updates `present` immediately (line 780)! The infrastructure for incremental updates exists. We're just not recomputing `total_cost` frequently enough.

---

## 🚨 WHY PREVIOUS EXPERIMENTS FAILED

### Experiment 1: Micro-batch (batch=8) with cost updates

**What we did:**
- Set MICRO_BATCH_SIZE = 8
- Added cost updates between batches
- Expected: Better negotiation

**Result:** **WORSE** (43 nets vs 182 baseline in iteration 1)

**Why it failed:**
```
Batch 1: 6/8 routed (75%) - costs reasonable
Batch 2: 4/8 routed (50%) - costs rising
...
Batch 8: 0/8 routed (0%)   ⚠️ Costs too high!
Batch 10: 0/8 routed (0%)  ⚠️ Blocked!
```

**Diagnosis:** Cost updates were too aggressive. With `pres_fac=1.0` in iteration 1, each batch adds pressure. By batch 8, costs are so high that NO paths exist.

**Lesson:** Cost updates need **damping** or **lower initial pres_fac** (e.g., 0.5 in iter 1, 1.0 in iter 2+).

### Experiment 2: Sequential (batch=1) without cost updates

**Result:** TERRIBLE (8 nets by iteration 30 vs 258 baseline)

**Why:** 512 GPU kernel launches with no negotiation benefit = all overhead, no gain.

### Experiment 3: Sequential (batch=1) WITH cost updates

**Result:** CRASH (8.6M log lines, stuck in iteration 1)

**Why:** 512 cost updates × verbose logging = explosion. Also, costs likely over-penalized like Experiment 1.

---

## ✅ THE CORRECT APPROACH

### Key Realizations

1. **Cost updates must be controlled**
   - Don't update after EVERY micro-batch in iteration 1
   - Use lower pres_fac in early iterations (0.5 instead of 1.0)
   - Or update every N batches, not every batch

2. **Batch size sweet spot is 16-32 nets**
   - Not 1 (too much overhead)
   - Not 8 (too aggressive with updates)
   - Not 128 (too little negotiation)

3. **Iteration 1 should be greedy**
   - Use large batches or no cost updates in iter 1
   - Reserve micro-batch negotiation for iterations 2+

### Recommended Implementation Strategy

**Hybrid approach: Iteration-dependent batching**

```python
def _route_all(self, tasks, all_tasks, pres_fac):
    ordered_nets = self._order_nets_by_difficulty(tasks)
    total = len(ordered_nets)

    # ITERATION 1: Greedy routing with large batches (fast initial routing)
    if self.iteration == 1:
        # Compute costs once (original behavior)
        self.accounting.update_costs(...)
        costs_frozen = self.accounting.total_cost.get()
        # Route all nets in large batches (fast)
        return self._route_all_batched_gpu(ordered_nets, ..., costs_frozen, ...)

    # ITERATION 2+: Micro-batch negotiation (quality routing)
    else:
        MICRO_BATCH_SIZE = 16
        routed_total = 0

        for batch_start in range(0, total, MICRO_BATCH_SIZE):
            batch_nets = ordered_nets[batch_start:batch_start+MICRO_BATCH_SIZE]

            # Recompute costs for this micro-batch
            self.accounting.update_costs(...)
            costs_live = self.accounting.total_cost.get()

            # Route micro-batch
            routed, failed = self._route_batch_gpu(batch_nets, ..., costs_live, ...)
            routed_total += routed

        return routed_total, total - routed_total
```

**Why this works:**
- Iteration 1 seeds 35-45% of nets quickly (baseline behavior)
- Iterations 2+ use micro-batch negotiation to resolve conflicts
- Later iterations see the benefit of live cost updates
- Avoids iteration 1 cost explosion problem

---

## 📝 STEP-BY-STEP IMPLEMENTATION

### Phase 1: Proof of Concept (30 minutes)

**Goal:** Prove CPU sequential routing works

**Change:**
```python
# File: unified_pathfinder.py
# Line: 2732

# BEFORE:
use_gpu_batching = (hasattr(self.solver, 'gpu_solver') and ...)

# AFTER:
use_gpu_batching = False  # Force CPU sequential for proof of concept
```

**Test:**
```bash
python main.py --test-manhattan > test_cpu_sequential.log 2>&1 &
# Wait 5-10 minutes (slow!)
grep "ITER [0-9].*routed=" test_cpu_sequential.log
# Expected: 80%+ success rate
```

**If this achieves 70%+ success:** Sequential routing is confirmed as THE fix. Proceed to Phase 2.

**If this still fails:** Problem is elsewhere (ROI, escape planner, base costs). STOP and investigate.

---

### Phase 2: Hybrid Implementation (Day 1-2)

**Step 1: Add micro_batch_size config** (15 minutes)

File: `orthoroute/algorithms/manhattan/pathfinder/config.py`

Find the `@dataclass class PathFinderConfig` and add:
```python
micro_batch_size: int = 16  # Batch size for iterations 2+ (lower = better negotiation)
```

**Step 2: Refactor `_route_all()`** (3-4 hours)

File: `unified_pathfinder.py:2706-2750`

**Current code:**
```python
def _route_all(self, tasks, all_tasks, pres_fac):
    # ... setup ...

    # Line 2727: Compute costs ONCE
    self.accounting.update_costs(...)
    costs = self.accounting.total_cost.get()

    # Line 2736: GPU batch all nets
    if use_gpu_batching:
        return self._route_all_batched_gpu(ordered_nets, tasks, all_tasks, costs, ...)

    # Line 2740: CPU fallback (already sequential)
    for net_id in ordered_nets:
        # ... sequential routing ...
```

**New code:**
```python
def _route_all(self, tasks, all_tasks, pres_fac):
    ordered_nets = self._order_nets_by_difficulty(tasks)
    total = len(ordered_nets)
    cfg = self.config

    # ITERATION 1: Greedy routing (fast, large batches)
    if self.iteration == 1:
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs = self.accounting.total_cost.get()

        use_gpu = hasattr(self.solver, 'gpu_solver') and self.solver.gpu_solver
        if use_gpu:
            return self._route_all_batched_gpu_legacy(ordered_nets, tasks, all_tasks, costs, pres_fac)
        else:
            return self._route_all_sequential_cpu(ordered_nets, tasks, all_tasks, costs)

    # ITERATION 2+: Micro-batch negotiation
    else:
        return self._route_all_microbatch(ordered_nets, tasks, all_tasks, pres_fac)


def _route_all_microbatch(self, ordered_nets, tasks, all_tasks, pres_fac):
    """Micro-batch routing with cost updates between batches (iterations 2+)"""
    cfg = self.config
    total = len(ordered_nets)
    MICRO_BATCH_SIZE = cfg.micro_batch_size

    routed_total = 0
    failed_total = 0
    batch_num = 0

    logger.info(f"[MICRO-BATCH] Routing {total} nets in batches of {MICRO_BATCH_SIZE} with cost updates")

    for batch_start in range(0, total, MICRO_BATCH_SIZE):
        batch_num += 1
        batch_end = min(batch_start + MICRO_BATCH_SIZE, total)
        batch_nets = ordered_nets[batch_start:batch_end]
        K = len(batch_nets)

        # ✅ RECOMPUTE COSTS FOR THIS MICRO-BATCH
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs_live = self.accounting.total_cost.get()

        logger.info(f"[MICRO-BATCH-{batch_num}] Routing nets {batch_start+1}-{batch_end}/{total} (K={K})")

        # Route micro-batch with GPU
        routed, failed = self._route_batch_gpu_single(
            batch_nets, tasks, all_tasks, costs_live, pres_fac
        )

        routed_total += routed
        failed_total += failed

        logger.info(f"[MICRO-BATCH-{batch_num}] Complete: {routed}/{K} routed")

    return routed_total, failed_total
```

**Step 3: Rename and simplify batch function** (2-3 hours)

File: `unified_pathfinder.py:2964-3600`

- Rename: `_route_all_batched_gpu` → `_route_all_batched_gpu_legacy` (keep for iter 1)
- Create new: `_route_batch_gpu_single()` (simplified version)
  - Input: Single batch_nets list (not all nets)
  - Input: costs parameter (already has this)
  - Remove: Outer loop over batches (lines 3061-3600)
  - Keep: All ROI extraction, GPU dispatch, result processing

**Step 4: Test** (2-4 hours)

```bash
# Test with micro_batch_size=16
python main.py --test-manhattan > test_microbatch_16.log 2>&1

# Compare to baseline
grep "ITER [0-9].*routed=" test_baseline_stock.log > baseline_results.txt
grep "ITER [0-9].*routed=" test_microbatch_16.log > microbatch_results.txt

# Success criteria:
# - Iteration 1: ~180-200 routed (similar to baseline, using large batches)
# - Iteration 5: >300 routed (vs baseline ~260)
# - Iteration 10: >400 routed (vs baseline ~280)
# - Final: 70%+ routed (vs baseline ~50%)
```

---

## 🧪 TESTING & VALIDATION

### Test Matrix

| Test ID | Config | Expected Result | Purpose |
|---------|--------|-----------------|---------|
| **T1** | CPU fallback (use_gpu=False) | 80%+ success | Prove sequential works |
| **T2** | Micro-batch size=32 | 60-70% success | Conservative |
| **T3** | Micro-batch size=16 | 70-80% success | **Recommended** |
| **T4** | Micro-batch size=8 | 75-85% success | Aggressive (may be slower) |
| **T5** | Hybrid (iter1=large, iter2+=micro) | 75-85% success | **Production candidate** |

### Validation Checklist

**Small board (256 nets):**
- [ ] Iteration 1: 35-40% routed (greedy baseline)
- [ ] Iteration 5: 55-65% routed (micro-batch starting to work)
- [ ] Iteration 10: 70-80% routed (negotiation in full effect)
- [ ] Final: 80-95% routed with low overuse
- [ ] GUI shows: "All nets routed with zero overuse" (screenshot)
- [ ] Log shows: `[CLEAN] All nets routed with zero overuse` (line 2385)

**Large board (8192 nets):**
- [ ] Iteration 1: 30-40% routed (baseline: 35%)
- [ ] Iteration 10: >1000 routed (baseline: stuck at 375)
- [ ] Iteration 30: 60-75% routed (baseline: 4.6%)
- [ ] No stagnation (routed count increases monotonically)

### Performance Expectations

**Small board (512 nets):**
| Method | Iter 1 Time | Total Time (30 iters) | Final Success |
|--------|-------------|----------------------|---------------|
| Baseline | 17s | ~3 min | 50% |
| Micro-batch (16) | 25-30s | ~8-12 min | **Target: 75-85%** |
| CPU Sequential | 60-120s | ~30-60 min | **Target: 85-95%** |

**Large board (8192 nets):**
| Method | Iter 1 Time | Estimated Total | Final Success |
|--------|-------------|-----------------|---------------|
| Baseline | ~2-3 min | ~60-90 min | 4.6% |
| Micro-batch (16) | ~8-12 min | **~4-6 hours** | **Target: 60-75%** |

---

## ⚙️ CONFIGURATION TUNING

### Micro-Batch Size Selection

**Formula:** Larger batch = faster but weaker negotiation

| Batch Size | Updates/Iter | GPU Calls | Speed | Negotiation Strength | Use Case |
|------------|--------------|-----------|-------|---------------------|----------|
| 1 | 512 | 512 | Very slow | Maximum | Research/proof |
| 8 | 64 | 64 | Slow | Strong | Dense boards |
| **16** | 32 | 32 | Moderate | **Good** | **Recommended** |
| 32 | 16 | 16 | Fast | Weak | Large sparse boards |
| 64 | 8 | 8 | Very fast | Minimal | Time-constrained |

### Pressure Factor Schedule

**Current (aggressive):**
```
Iter 1: pres_fac = 1.0
Iter 2: pres_fac = 2.0
Iter 3: pres_fac = 4.0
...
```

**Recommended for micro-batch:**
```python
# Lower initial pressure to avoid over-penalizing
if self.iteration == 1:
    pres_fac = 0.5  # Gentler start
elif self.iteration <= 5:
    pres_fac = 1.0 * (1.5 ** (self.iteration - 1))  # Gentler ramp
else:
    pres_fac = 1.0 * (2.0 ** (self.iteration - 1))  # Standard PathFinder
```

### History Decay

**Current:** `history *= 0.98` before adding increment (line 826)

**Recommendation:** Keep as-is. Decay helps prevent cost explosion in long runs.

---

## 🎯 IMPLEMENTATION ROADMAP

### Week 1: Core Refactor

**Day 1: Proof of Concept**
- [ ] Test CPU fallback (use_gpu=False)
- [ ] Verify sequential achieves 80%+ success
- [ ] Document results

**Day 2: Micro-Batch Implementation**
- [ ] Add `micro_batch_size` to config.py
- [ ] Refactor `_route_all()` with micro-batch loop
- [ ] Create `_route_all_microbatch()` function
- [ ] Simplify `_route_batch_gpu_single()`

**Day 3: Testing & Debugging**
- [ ] Test micro_batch_size = 8, 16, 32
- [ ] Tune pressure factor schedule
- [ ] Fix any accounting bugs
- [ ] Validate on small board

**Day 4: Hybrid Iteration Strategy**
- [ ] Implement iter 1 = large batch, iter 2+ = micro-batch
- [ ] Test on small board
- [ ] Measure performance regression

**Day 5: Large Board Validation**
- [ ] Run large board test with micro-batch
- [ ] Monitor for stagnation (should be gone)
- [ ] Verify routing quality improves across iterations
- [ ] Benchmark performance

### Week 2: Optimization & Cleanup

**Day 6-7: Performance Tuning**
- [ ] Profile micro-batch overhead
- [ ] Optimize ROI extraction (reduce logging)
- [ ] Consider batch consolidation strategies

**Day 8-10: Advanced Features**
- [ ] Adaptive batch sizing (start large, shrink on conflicts)
- [ ] Net ordering strategies (route easy nets first to build corridors)
- [ ] Freeze successful nets to reduce hotset size

---

## 📐 DETAILED CODE CHANGES

### Change 1: Add Config Parameter

**File:** `orthoroute/algorithms/manhattan/pathfinder/config.py`

**Find:**
```python
@dataclass
class PathFinderConfig:
    max_iterations: int = 30
    batch_size: int = 32
    # ... other params ...
```

**Add after batch_size:**
```python
    micro_batch_size: int = 16  # Micro-batch size for iterations 2+ (negotiation mode)
    use_micro_batch_negotiation: bool = True  # Enable micro-batch negotiation
```

---

### Change 2: Refactor _route_all()

**File:** `unified_pathfinder.py`
**Lines:** 2706-2750

**BEFORE (current):**
```python
def _route_all(self, tasks: Dict[str, Tuple[int, int]], all_tasks: Dict[str, Tuple[int, int]] = None, pres_fac: float = 1.0) -> Tuple[int, int]:
    """Route nets with adaptive ROI extraction and intra-iteration cost updates"""
    if all_tasks is None:
        all_tasks = tasks

    routed_this_pass = 0
    failed_this_pass = 0
    total = len(tasks)
    cfg = self.config

    # ... setup code (lines 2716-2723) ...

    # Compute costs once per iteration (not per net) - major performance win
    # Note: via_cost_multiplier and base_cost_weight are applied here
    self.accounting.update_costs(self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
                                 base_cost_weight=cfg.base_cost_weight)
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
        # ... CPU sequential routing ...
```

**AFTER (micro-batch negotiation):**
```python
def _route_all(self, tasks: Dict[str, Tuple[int, int]], all_tasks: Dict[str, Tuple[int, int]] = None, pres_fac: float = 1.0) -> Tuple[int, int]:
    """Route nets with micro-batch negotiation (iter 1: greedy, iter 2+: sequential)"""
    if all_tasks is None:
        all_tasks = tasks

    total = len(tasks)
    cfg = self.config

    # Reset full-graph fallback counter at start of iteration
    self.full_graph_fallback_count = 0

    # ROI margin grows with stagnation
    roi_margin_bonus = self.stagnation_counter * 0.6

    # Order nets by difficulty
    ordered_nets = self._order_nets_by_difficulty(tasks)

    # Check if GPU available
    use_gpu = hasattr(self.solver, 'gpu_solver') and self.solver.gpu_solver is not None

    # ITERATION 1: Greedy routing with large batches (fast initial coverage)
    if self.iteration == 1:
        # Compute costs once for iteration 1 (original behavior)
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost

        if use_gpu and total > 8:
            logger.info(f"[ITER-1-GREEDY] Routing {total} nets with large batches (fast mode)")
            return self._route_all_batched_gpu(ordered_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus)
        else:
            return self._route_all_sequential_cpu(ordered_nets, tasks, all_tasks, pres_fac)

    # ITERATION 2+: Micro-batch negotiation (quality mode)
    elif cfg.use_micro_batch_negotiation:
        logger.info(f"[ITER-{self.iteration}-MICRO] Using micro-batch negotiation mode")
        return self._route_all_microbatch(ordered_nets, tasks, all_tasks, pres_fac, roi_margin_bonus, use_gpu)

    # Fallback to legacy behavior
    else:
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost

        if use_gpu and total > 8:
            return self._route_all_batched_gpu(ordered_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus)
        else:
            return self._route_all_sequential_cpu(ordered_nets, tasks, all_tasks, pres_fac)
```

---

### Change 3: Add `_route_all_microbatch()` Method

**File:** `unified_pathfinder.py`
**Location:** After `_route_all()` (around line 2960)

**New function (insert):**
```python
def _route_all_microbatch(self, ordered_nets, tasks, all_tasks, pres_fac, roi_margin_bonus, use_gpu):
    """
    Micro-batch PathFinder negotiation (iterations 2+).

    Routes nets in small batches (K=8-32) with cost recomputation between batches.
    This restores PathFinder's negotiation property: later nets see pressure from earlier nets.

    Args:
        ordered_nets: List of net IDs in routing order
        tasks: Dict of net_id -> (src, dst)
        all_tasks: Dict of all nets (for context)
        pres_fac: Pressure factor for current iteration
        roi_margin_bonus: ROI expansion bonus from stagnation
        use_gpu: Whether GPU is available

    Returns:
        (routed_count, failed_count)
    """
    import time
    cfg = self.config
    total = len(ordered_nets)
    MICRO_BATCH_SIZE = cfg.micro_batch_size

    routed_total = 0
    failed_total = 0
    batch_num = 0
    overall_start = time.time()

    logger.info(f"[MICRO-BATCH-NEGOTIATION] Routing {total} nets in batches of {MICRO_BATCH_SIZE}")
    logger.info(f"[MICRO-BATCH-NEGOTIATION] Will perform {(total + MICRO_BATCH_SIZE - 1) // MICRO_BATCH_SIZE} cost updates")

    for batch_start in range(0, total, MICRO_BATCH_SIZE):
        batch_num += 1
        batch_end = min(batch_start + MICRO_BATCH_SIZE, total)
        batch_nets = ordered_nets[batch_start:batch_end]
        K = len(batch_nets)

        batch_start_time = time.time()

        # ✅ CRITICAL: Recompute costs for THIS micro-batch
        # This allows later batches to see pressure from earlier batches
        self.accounting.update_costs(
            self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
            base_cost_weight=cfg.base_cost_weight
        )
        costs_live = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost

        logger.info(f"[MICRO-BATCH-{batch_num}/{(total + MICRO_BATCH_SIZE - 1) // MICRO_BATCH_SIZE}] Routing nets {batch_start+1}-{batch_end}/{total} (K={K})")

        # Route micro-batch
        if use_gpu:
            routed, failed = self._route_batch_gpu_single(
                batch_nets, tasks, all_tasks, costs_live, pres_fac, roi_margin_bonus
            )
        else:
            routed, failed = self._route_batch_cpu(batch_nets, tasks, costs_live)

        routed_total += routed
        failed_total += failed

        batch_time = time.time() - batch_start_time
        logger.info(f"[MICRO-BATCH-{batch_num}] Complete: {routed}/{K} routed in {batch_time:.2f}s")

    overall_time = time.time() - overall_start
    logger.info(f"[MICRO-BATCH-SUMMARY] {routed_total}/{total} routed in {overall_time:.2f}s ({total/overall_time:.1f} nets/sec)")

    return routed_total, failed_total
```

---

### Change 4: Create `_route_batch_gpu_single()` (Simplified Batch Function)

**File:** `unified_pathfinder.py`
**Location:** After `_route_all_microbatch()` (around line 3050)

**Strategy:** Copy-paste `_route_all_batched_gpu` (lines 2964-3600) and simplify:

**Simplifications needed:**
1. Remove outer batch loop (lines 3061-3600) - caller provides single batch
2. Remove batch size calculation (lines 2971-3028) - use len(batch_nets)
3. Remove short/long net separation (lines 3042-3071) - already ordered by caller
4. Keep all ROI extraction logic (lines 3078-3350)
5. Keep GPU dispatch (line 3409)
6. Keep result processing (lines 3413-3553)

**Skeleton:**
```python
def _route_batch_gpu_single(self, batch_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus):
    """
    Route a single micro-batch of nets using GPU.

    Simplified version of _route_all_batched_gpu that processes ONE batch at a time.
    Caller is responsible for cost recomputation between batches.

    Args:
        batch_nets: List of net IDs for THIS batch (K=8-32)
        costs: Pre-computed costs for THIS batch (recomputed by caller)

    Returns:
        (routed, failed)
    """
    import numpy as np
    import time

    cfg = self.config
    K = len(batch_nets)

    # Copy CSR arrays once
    shared_indptr = self.graph.indptr.get() if hasattr(self.graph.indptr, 'get') else self.graph.indptr
    shared_indices = self.graph.indices.get() if hasattr(self.graph.indices, 'get') else self.graph.indices
    shared_weights = costs.get() if hasattr(costs, 'get') else costs

    # Build ROI batch (copy logic from lines 3078-3350)
    roi_batch = []
    for net_id in batch_nets:
        src, dst = tasks[net_id]

        # Clear old path if rerouting (lines 3095-3100)
        # ... (keep as-is)

        # Extract ROI (lines 3103-3157)
        # ... (keep as-is)

        # Build ROI CSR (lines 3218-3230)
        # ... (keep as-is)

        roi_batch.append((src, dst, roi_indptr, roi_indices, roi_weights, roi_size, ...))

    # Route batch with GPU (line 3409)
    paths = self.solver.gpu_solver.find_paths_on_rois(roi_batch, use_bitmap=(self.iteration > 1))

    # Process results (lines 3413-3553)
    routed = 0
    for i, net_id in enumerate(batch_nets):
        path = paths[i] if i < len(paths) else None
        if path and len(path) > 1:
            # Convert to global, validate, commit
            # ... (keep logic from lines 3458-3515)
            edges = self._path_to_edges(path)
            self.accounting.commit_path(edges)  # ✅ Updates present!
            self.net_paths[net_id] = path
            routed += 1

    return routed, K - routed
```

---

## 🧪 PROOF-OF-CONCEPT TEST (Phase 1)

Before spending days on refactoring, prove the concept works:

### Quick Test: Force CPU Sequential

**Modify:** `unified_pathfinder.py:2732`

```python
# BEFORE:
use_gpu_batching = (hasattr(self.solver, 'gpu_solver') and
                   self.solver.gpu_solver is not None and
                   total > 8)

# AFTER:
use_gpu_batching = False  # ← Force CPU sequential PathFinder
```

**Run:**
```bash
python main.py --test-manhattan > test_cpu_sequential_proof.log 2>&1 &

# Wait 10-30 minutes (CPU is slow)
tail -f test_cpu_sequential_proof.log | grep "ITER [0-9]"

# After completion:
grep "ITER [0-9].*routed=" test_cpu_sequential_proof.log
grep "All nets routed" test_cpu_sequential_proof.log
```

**Expected results if hypothesis is correct:**
- Iteration 5: 60-70% routed
- Iteration 10: 75-85% routed
- Final: 85-95% routed with low overuse
- Should see "All nets routed with zero overuse"

**If CPU sequential achieves 80%+:** Hypothesis confirmed. Proceed with micro-batch refactor.

**If CPU sequential also fails:** Problem is elsewhere (ROI extraction, escape planner, cost formula). Need different investigation.

---

## 📊 DATA FROM PREVIOUS TESTS

### Test: Baseline (Stock) - 2025-10-24 10:11-10:14

**Configuration:**
- Large batches (effective_batch_size ~100-512)
- Costs computed once per iteration
- Full-graph ROI (no L-corridor)

**Results:**
```
Iter 1:  182/512 routed (35.5%) | overuse=1315 | time=17s
Iter 2:  221/512 routed (43.2%) | overuse=1596
Iter 3:  239/512 routed (46.7%) | overuse=1609
Iter 4:  252/512 routed (49.2%) | overuse=1748
Iter 5:  262/512 routed (51.2%) | overuse=1795
Iter 6:  266/512 routed (51.9%) | overuse=1763
Iter 7:  254/512 routed (49.6%) | overuse=972
Iter 8:  268/512 routed (52.3%) | overuse=1063
Iter 9:  273/512 routed (53.3%) | overuse=1025
Iter 10: 280/512 routed (54.7%) | overuse=1061
...
Iter 20: 268/512 routed (52.3%) | overuse=439
Iter 28: 258/512 routed (50.4%) | overuse=329
Iter 29: 258/512 routed (50.4%) | overuse=329  ⚠️ Stuck
Iter 30: 258/512 routed (50.4%) | overuse=335  ⚠️ Stuck

Total time: ~3 minutes for 30 iterations
```

**Observations:**
- Improves 36% → 55% from iter 1 to iter 10
- **Plateaus around 50-52%** and cannot improve further
- Gets stuck in iterations 28-30 (same 258 nets)
- Fast but limited quality

---

### Test: Large Board - Previous Analysis

**Configuration:**
- 8192 nets (vs 512 on small board)
- Same large-batch frozen-cost approach

**Results:**
```
Iteration 1-30: 375/8192 routed (4.6%) ⚠️ STUCK
Every iteration: Same 375 nets succeed
Pressure factor: 1.0 → 64.0 (no effect)
ROI: max_roi_size == N_global (2.47M nodes = full graph)
```

**Root causes identified:**
1. **Frozen costs** - All 8192 nets route with same cost snapshot
2. **Full-graph ROI** - No spatial locality, all nets compete globally
3. **Massive parallelism** - 8192 nets / 512 batch = 16 batches with no negotiation

---

## 🤔 DOUBLE-CHECK: IS THIS THE RIGHT APPROACH?

### Evidence Supporting Sequential Routing

**1. Literature:**
- PathFinder paper (McMurchie & Ebeling 1995) describes sequential routing
- VPR (industry-standard FPGA router) implements PathFinder sequentially
- No "parallel PathFinder" in production routers

**2. Symptoms match frozen-cost hypothesis:**
- Vertical band gaps → all nets pick same "cheap" path
- Stagnation → same nets succeed because they route first with lowest costs
- Large board collapse → problem magnifies with scale (more nets = more collisions)

**3. EdgeAccountant already supports it:**
- `commit_path()` updates both canonical AND present (line 775-780)
- `update_costs()` reads from present (line 845)
- Infrastructure exists for live updates

**4. CPU fallback path exists:**
- Lines 2740-2900 already implement sequential routing
- If CPU achieves high success, that confirms the approach

### Evidence Against / Concerns

**1. Micro-batch experiments failed:**
- Batch=8 achieved only 43 nets (vs 182 baseline)
- Many batches had 0/8 success rate
- **Counter-argument:** We didn't tune pres_fac - it was too aggressive

**2. Performance cost:**
- 3-10× slower depending on batch size
- May not be acceptable for large boards
- **Counter-argument:** Correctness > speed; can optimize later

**3. GPU underutilization:**
- GPU designed for massive parallelism
- Micro-batches of 16 don't fully utilize GPU
- **Counter-argument:** Better than no solution; can investigate dynamic cost kernel later

### Verdict: YES, This is the Right Approach

**Confidence level: 85%**

The evidence strongly supports that frozen costs break negotiation. The failed micro-batch experiments were likely due to parameter tuning (too-aggressive pres_fac), not fundamental architectural issues.

**Recommended next steps:**
1. **Phase 1 (proof):** Test CPU fallback - if it achieves 80%+, we're on the right track
2. **Phase 2 (implement):** Micro-batch with TUNED parameters (lower pres_fac, larger batch size)
3. **Phase 3 (optimize):** Performance tuning once correctness is proven

---

## 🎬 NEXT STEPS FOR IMPLEMENTATION

### Immediate Actions (This Session)

**Action 1: Proof-of-concept test**
```bash
# Edit unified_pathfinder.py:2732
use_gpu_batching = False  # Force CPU

# Run test
python main.py --test-manhattan > test_cpu_proof.log 2>&1 &

# Monitor (will take 10-30 min)
watch -n 10 "grep 'ITER [0-9].*routed=' test_cpu_proof.log | tail -5"
```

**If CPU achieves 70%+:** Green light for micro-batch refactor

**If CPU also fails:** Red flag - investigate other issues (ROI, escape planner)

---

### Next Session Tasks

**Task 1: Implement hybrid iteration strategy**
- Iteration 1: Keep large batches (fast greedy routing)
- Iteration 2+: Micro-batch negotiation

**Task 2: Tune pressure factor**
- Start lower (pres_fac=0.5 in iter 1)
- Ramp more gently (1.5× multiplier instead of 2.0×)

**Task 3: Add batch size parameter**
- Make MICRO_BATCH_SIZE configurable
- Test 8, 16, 32 to find sweet spot

**Task 4: Reduce logging verbosity**
- Only log batch summaries, not every net
- Prevent log explosion like in batch=1 test

---

## 📈 SUCCESS METRICS

### Small Board (512 nets)

**Current baseline:**
- Iter 1: 182 routed (35.5%)
- Iter 10: 280 routed (54.7%)
- Final: 258 routed (50.4%)
- Time: ~3 minutes

**Target with micro-batch:**
- Iter 1: 180-200 routed (35-40%) - similar to baseline
- Iter 10: 350-400 routed (68-78%) - **+70-120 nets improvement**
- Final: 400-480 routed (78-94%) - **+140-220 nets improvement**
- Time: 8-15 minutes (acceptable)
- Should see "All nets routed with zero overuse"

### Large Board (8192 nets)

**Current baseline:**
- All iterations: 375 routed (4.6%) ⚠️ STUCK
- No improvement across 30 iterations
- ROI = full graph (no locality)

**Target with micro-batch:**
- Iter 1: 2500-3500 routed (30-43%)
- Iter 10: 5000-6000 routed (61-73%)
- Iter 30: 6000-7000 routed (73-85%)
- No stagnation (steady improvement)
- Overuse decreases over time

---

## 🔬 TECHNICAL DEEP DIVE

### The Cost Freezing Problem

**Location:** `unified_pathfinder.py:2727-2729`

```python
# Line 2727: Costs computed ONCE per iteration
self.accounting.update_costs(self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
                             base_cost_weight=cfg.base_cost_weight)

# Line 2729: Costs frozen into variable
costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost

# Line 2738: Frozen costs passed to ALL batches
return self._route_all_batched_gpu(ordered_nets, tasks, all_tasks, costs, pres_fac, roi_margin_bonus)
                                                                      ^^^^^ This never changes!
```

**Inside `_route_all_batched_gpu` (lines 2964-3600):**
```python
# Line 3031: Costs from parameter (frozen)
shared_weights = costs.get() if hasattr(costs, 'get') else costs

# Line 3061: Outer batch loop
for batch_start in range(0, total, effective_batch_size):
    batch_nets = ordered_nets[batch_start:batch_end]

    # Lines 3082-3350: Build ROI batch using shared_weights (frozen!)
    for net_id in batch_nets:
        roi_weights = shared_weights  # ← Same frozen costs for all batches!
        roi_batch.append((..., roi_weights, ...))

    # Line 3409: GPU routes with frozen costs
    paths = self.solver.gpu_solver.find_paths_on_rois(roi_batch)

    # Lines 3508-3515: Commit paths
    # present[edges] updated here, but costs not recomputed!
```

**The cycle of failure:**
- Batch 1 routes → commits paths → present[edges] increases
- Batch 2 routes → **still uses old costs** → picks same edges
- Batch 3 routes → **still uses old costs** → picks same edges
- All batches dogpile the same corridors
- Next iteration: costs recomputed, but damage already done

### Why CPU Fallback Might Work

**Location:** `unified_pathfinder.py:2740-2900`

The CPU path routes sequentially but **may or may not update costs between nets** - need to verify!

Looking at lines 2758-2900, the CPU fallback:
- Routes one net at a time (sequential)
- Commits paths immediately (updates present)
- BUT: Uses frozen `costs` from line 2729!

So CPU fallback has the SAME frozen cost problem! This explains why we need to test it - it's not clear it will work either.

**ACTION ITEM:** Check if CPU fallback recomputes costs between nets or also uses frozen costs.

---

## 🐛 KNOWN ISSUES TO WATCH FOR

### Issue 1: Cost Explosion

**Symptom:** Later batches have 0% success rate

**Cause:** Cost updates build up pressure too fast:
```
Batch 1: present[edges] = 1-2
Batch 2: present[edges] = 3-5  → costs spike
Batch 8: present[edges] = 15-20 → costs infinite
```

**Solution:**
- Use lower initial pres_fac (0.5 instead of 1.0)
- Update costs every N batches, not every batch
- Scale pres_fac by iteration number

### Issue 2: Log Explosion

**Symptom:** Multi-million line logs, test runs for hours

**Cause:** Verbose logging × cost updates × batch count

**Solution:**
- Only log batch summaries, not individual nets
- Use logger.debug() for per-net messages
- Reduce GPU kernel logging

### Issue 3: Accounting Drift

**Symptom:** present != canonical after iteration

**Cause:** Race condition or missing sync in commit_path()

**Solution:**
- Call `verify_present_matches_canonical()` after each iteration (already exists!)
- Check line 2352-2353 for mismatch warnings

---

## 📂 FILE-BY-FILE CHANGE CHECKLIST

### ✅ unified_pathfinder.py

**Function: `_route_all()` (lines 2706-2750)**
- [x] Remove global cost computation (lines 2727-2729) for iter 2+
- [x] Add iteration check: if iter==1 use large batches, else micro-batch
- [x] Add micro-batch loop with cost updates

**Function: `_route_all_batched_gpu()` (lines 2964-3600)**
- [x] Keep for iteration 1 (rename to `_route_all_batched_gpu_legacy`)
- [x] Create new `_route_batch_gpu_single()` for micro-batches
- [x] Remove outer batch loop from new function

**Function: `_pathfinder_negotiation()` (lines 2286-2530)**
- [ ] No changes needed (already calls `_route_all()`)

### ✅ config.py

**Add parameters:**
```python
@dataclass
class PathFinderConfig:
    # ... existing params ...

    # Micro-batch negotiation (iterations 2+)
    use_micro_batch_negotiation: bool = True
    micro_batch_size: int = 16  # Sweet spot: 8-32

    # Gentler pressure schedule for micro-batch mode
    micro_batch_initial_pres_fac: float = 0.5  # Start lower
    micro_batch_pres_fac_mult: float = 1.5  # Gentler ramp (vs 2.0 baseline)
```

### ✅ cuda_dijkstra.py

**No changes required!**

The GPU kernel already handles arbitrary K (1 to 512+). Micro-batches with K=16 work fine.

### ✅ Other files

**No changes needed:**
- EdgeAccountant (already incremental)
- ROI extraction (already per-net)
- Graph builder (already supports any batch size)
- Negotiation mixin (hotset logic unchanged)

---

## 🎯 EXPECTED TIMELINE

### Conservative Estimate (Full Sequential)

**Day 1:** Proof of concept
- Test CPU fallback
- Validate hypothesis
- **Deliverable:** Evidence that sequential works

**Day 2:** Core refactor
- Implement micro-batch loop in `_route_all()`
- Create `_route_batch_gpu_single()`
- **Deliverable:** Code compiles and runs

**Day 3:** Testing & tuning
- Test batch sizes 8, 16, 32
- Tune pressure factor schedule
- **Deliverable:** Small board achieves 75%+ success

**Day 4:** Large board validation
- Run large board test
- Monitor for stagnation
- **Deliverable:** Large board >60% success

**Day 5:** Optimization & cleanup
- Profile performance
- Reduce logging
- **Deliverable:** Production-ready code

### Optimistic Estimate (If CPU Proof Works Well)

**Day 1:** Proof + implementation
- Morning: CPU proof (4 hours)
- Afternoon: Micro-batch refactor (4 hours)

**Day 2:** Testing & validation
- Test all batch sizes
- Validate both boards
- **Deliverable:** Production code

---

## 🚀 FINAL RECOMMENDATIONS

### Recommended Path Forward

1. **START HERE:** Test CPU fallback (30 min effort, high value)
   - Proves/disproves hypothesis
   - No risk (easily reverted)

2. **If CPU works:** Implement Option B (Micro-Batch Hybrid)
   - Iteration 1: Large batches (fast)
   - Iteration 2+: Micro-batches (quality)
   - Tunable parameter

3. **If CPU fails:** Investigate other root causes
   - ROI extraction (falling back to full graph?)
   - Pad escape planner (creating systematic conflicts?)
   - Cost formula (base_cost_weight too low?)

### Configuration Recommendations

**Conservative (prioritize correctness):**
```python
micro_batch_size = 8
micro_batch_initial_pres_fac = 0.3
micro_batch_pres_fac_mult = 1.3
```

**Balanced (recommended):**
```python
micro_batch_size = 16
micro_batch_initial_pres_fac = 0.5
micro_batch_pres_fac_mult = 1.5
```

**Aggressive (prioritize speed):**
```python
micro_batch_size = 32
micro_batch_initial_pres_fac = 0.7
micro_batch_pres_fac_mult = 1.8
```

---

## 📸 VALIDATION: The "All Nets Routed" Screenshot

The screenshot showing "Routing completed successfully! Results: 2063 tracks placed, 1037 vias placed. All nets routed with zero overuse" is the gold standard.

**This message comes from:** `unified_pathfinder.py:2385`
```python
# Clean-phase: if overuse==0, freeze good nets and finish stragglers
if over_sum == 0:
    unrouted = {nid for nid in tasks.keys() if not self.net_paths.get(nid)}
    if not unrouted:
        logger.info("[CLEAN] All nets routed with zero overuse")  # ← This!
```

**To achieve this:**
1. All 512 nets must have paths (`self.net_paths[nid]` for all nid)
2. Overuse must be zero (`over_sum == 0`)
3. Must happen before max_iterations reached

**Current baseline never achieves this** - it plateaus at 50% with persistent overuse.

**Target:** Micro-batch negotiation should achieve this by iteration 15-25.

---

## 🔧 TROUBLESHOOTING GUIDE

### Problem: Micro-batch still fails (low success rate)

**Check:**
1. Is pres_fac too high? (Try 0.3-0.5 for iteration 1)
2. Are costs being recomputed? (Add logging in cost update)
3. Is present being updated? (Check accounting.verify_present_matches_canonical())

### Problem: Test takes too long

**Check:**
1. Is logging too verbose? (Reduce to summaries only)
2. Is batch size too small? (Try 32 instead of 8)
3. Is GPU kernel launching efficiently? (Check batch preparation time)

### Problem: Routing gets worse across iterations

**Check:**
1. Is history accumulating correctly? (Should have decay)
2. Is pres_fac ramping too fast? (Use gentler multiplier)
3. Are nets oscillating? (Check if same nets keep failing)

---

## 🏁 SUCCESS CRITERIA

**Phase 1 complete when:**
- [ ] CPU fallback test achieves 70%+ success
- [ ] Confirmed that sequential routing works
- [ ] Decision made to proceed with micro-batch refactor

**Phase 2 complete when:**
- [ ] Code refactored to micro-batch architecture
- [ ] Small board achieves 75%+ success
- [ ] No accounting errors
- [ ] Tests pass consistently

**Phase 3 complete when:**
- [ ] Large board achieves 60%+ success
- [ ] Performance acceptable (<6 hours for large board)
- [ ] "All nets routed with zero overuse" achieved on small board
- [ ] Production ready

---

## 📞 QUESTIONS FOR HUMAN

1. **Performance vs Quality trade-off:** Is 5-10× slower acceptable for 2× better routing quality?

2. **Iteration 1 strategy:** Keep fast large batches for greedy seeding, or use micro-batch from start?

3. **Target platform:** Is this for PCB production (quality critical) or rapid prototyping (speed critical)?

4. **Large board priority:** Is solving the 8192-net large board the primary goal?

---

**Last updated:** 2025-10-24
**Status:** Analysis complete, ready for implementation
**Recommended next action:** Run CPU fallback proof-of-concept test


---

## ⚠️ CRITICAL UPDATE: CPU Fallback Investigation

**Date:** 2025-10-24

**Finding:** The CPU fallback code (lines 2740-2900) ALSO uses frozen costs from line 2729!

**Evidence:**
- Line 2729: `costs = self.accounting.total_cost.get()` (frozen for iteration)
- Line 2884: `self.solver.find_path_multisource_multisink(..., costs, ...)` (uses frozen costs)
- Line 2894: `self.solver.find_path_roi(src, dst, costs, ...)` (uses frozen costs)

**Implication:** Testing CPU fallback will NOT prove the hypothesis because it has the same architectural flaw.

**Action required:** Must implement micro-batch refactor directly, or modify CPU fallback to recompute costs per net.

**Quick fix for CPU proof-of-concept:**
```python
# Inside CPU loop (after line 2756)
# Add cost recomputation:
self.accounting.update_costs(
    self.graph.base_costs, pres_fac, cfg.hist_cost_weight,
    base_cost_weight=cfg.base_cost_weight
)
costs = self.accounting.total_cost.get() if self.accounting.use_gpu else self.accounting.total_cost
```

This would enable TRUE sequential PathFinder in CPU mode for testing.

