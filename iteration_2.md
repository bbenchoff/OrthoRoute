# Iteration 2: Fixed Portal Seed Preservation

## Changes from Iteration 1:

### 1. **Portal Seeds Now Passed to ROI Extraction**
The critical bug was that `extract_roi()` was only preserving the 2 entry_layer portal nodes, but routing actually needs ALL 12 layers at each portal location for multi-source/multi-sink Dijkstra.

**Fix:** Added code to get portal seeds and pass them to extract_roi:
```python
src_seeds = self._get_portal_seeds(src_portal)
dst_seeds = self._get_portal_seeds(dst_portal)
portal_seeds = src_seeds + dst_seeds

roi_nodes, global_to_roi = self.roi_extractor.extract_roi(
    src_portal_node, dst_portal_node, initial_radius=adaptive_radius,
    stagnation_bonus=roi_margin_bonus, portal_seeds=portal_seeds
)
```

This ensures ALL 24 portal nodes (12 src + 12 dst) are preserved in the truncated ROI.

### 2. **BFS Early Stopping**
Added two performance optimizations:
- **Max depth cap:** 500 steps (200mm @ 0.4mm grid) prevents unbounded BFS explosion
- **Early ROI size check:** Stop BFS when ROI exceeds 450K nodes (will be truncated to 300K anyway)

**Expected improvements:**
- ROI extraction should be 3-5x faster (2-3 sec instead of 8-10 sec)
- Sink nodes should be reachable (Sink_dist should be FINITE!)

## Test Started: 2025-10-08 09:XX:XX

Monitoring for:
1. Sink_dist: Should be finite (not inf)
2. ROI extraction speed: Target <3 sec per net
3. Paths found: Target >0 in iteration 1
4. Early stop messages: Should see BFS stopping at 450K or 500 steps

## Expected Results:
- **Sink reachability:** 100% finite distances
- **ROI extraction:** 2-3 seconds per net
- **Paths found:** >50% success in iteration 1
- **Total time:** ~15 minutes for first batch of 32 nets
