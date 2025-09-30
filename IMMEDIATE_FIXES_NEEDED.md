# Immediate Performance & Convergence Fixes

## Priority 1: Fix CSR Rebuild Spam (2 min savings)

**Problem:** CSR lookup rebuilds 6 times despite guard
**Root cause:** `E_live` attribute not set, guard always sees `E_live=0`

**Fix:**
```python
# In _build_gpu_matrices or wherever E_live should be set:
self.E_live = self.edge_present_usage.shape[0]

# The guard at line 6999 will then work:
current_E = getattr(self, 'E_live', 0)  # Will now be 13667040
```

**Expected:** ONE CSR build, saves ~2 minutes

## Priority 2: Speed Up Escape Routing (5 min savings)

**Problem:** 408s for 3200 pads = 7.8s per 100 pads
**Likely causes:**
- Creating escape stubs one at a time
- Repeated spatial index lookups
- Coordinate array extensions in tight loop

**Investigate:**
- `_connect_pads_optimized()` line ~1974
- `_create_escape_stub()` line ~2067
- Look for O(n²) loops or repeated array resizing

**Target:** <60s for 3200 pads (20x faster)

## Priority 3: Why Hotset Nets Can't Route

**Problem:** All 191 hotset nets fail at iters 3-6 despite:
- ROI widening (multiplier 2.20)
- Cost escalation (pres_fac 60.40)
- Bus layer spreading active

**Possible causes:**
1. **ROI extraction failure** - ROI is empty or doesn't include alternatives
2. **Cost infinity** - All alt paths have inf cost (hard locks?)
3. **True capacity problem** - Legitimately need >12 layers

**Debug needed:**
- Log ROI size for a failing hotset net
- Log cost distribution (how many edges at inf vs finite)
- Check if `_converged=False` is actually working

## Priority 4: Verify Cost Propagation to Solver

**Problem:** Costs update but routing doesn't change

**Check:**
- In `_cpu_dijkstra_roi_heap()` ~line 4546: Is it using `edge_total_cost`?
- In `_gpu_dijkstra_roi_csr()` ~line 4645: Is it using `edge_total_cost`?
- Are there other routing methods not updated?

**Target Log:**
```
[SOLVER-COST-CHECK] net=B10B14_001 avg_edge_cost=157.3 (base=1.0 pres=156.3)
```

## Expected Improvements

After fixes:
- **Startup:** 8 min → 1 min (eliminate 408s escape + 120s CSR spam)
- **Iteration time:** 70s → 20-30s (faster routing with proper costs)
- **Convergence:** Either routes succeed OR clear "need 16+ layers" message