# Δ-Stepping Algorithm Flow Visualization

**Visual guide to understand bucket-based priority queue expansion**

---

## BFS Wavefront (OLD) vs Δ-Stepping (NEW)

### OLD: BFS Wavefront Expansion (INCORRECT)

```
Iteration 0: Process ALL nodes in frontier (regardless of cost)
┌─────────────────────────────────────────────────┐
│ Frontier: [A:0, B:5, C:10, D:15, E:20, F:25]   │ ← Mixed costs!
│ Expand ALL 6 nodes in parallel                  │
└─────────────────────────────────────────────────┘
         │
         ▼
Iteration 1: Process ALL newly discovered nodes
┌─────────────────────────────────────────────────┐
│ Frontier: [G:8, H:12, I:18, J:22, K:30, L:35]  │ ← Still mixed!
│ Expand ALL 6 nodes in parallel                  │
└─────────────────────────────────────────────────┘

PROBLEM: Nodes with cost 35 processed BEFORE nodes with cost 8!
         This violates Dijkstra's correctness guarantee.
```

### NEW: Δ-Stepping Bucket Expansion (CORRECT)

```
Bucket 0 [0-0.5mm]:   Process all nodes with cost 0.0 - 0.5
┌─────────────────────────────────────────────────┐
│ Bucket 0: [A:0.0, B:0.4]                       │ ← Same cost range
│ Expand: A:0.0 → discovers [C:0.8, D:1.2]       │
│         B:0.4 → discovers [E:0.9, F:1.5]       │
└─────────────────────────────────────────────────┘
         │
         ▼ Insert into appropriate buckets
         C:0.8 → Bucket 1 [0.5-1.0mm]
         D:1.2 → Bucket 2 [1.0-1.5mm]
         E:0.9 → Bucket 1 [0.5-1.0mm]
         F:1.5 → Bucket 3 [1.5-2.0mm]

Bucket 1 [0.5-1.0mm]:  Process all nodes with cost 0.5 - 1.0
┌─────────────────────────────────────────────────┐
│ Bucket 1: [C:0.8, E:0.9]                       │
│ Expand: C:0.8 → discovers [G:1.3, H:1.6]       │
│         E:0.9 → discovers [I:1.4, J:1.8]       │
└─────────────────────────────────────────────────┘
         │
         ▼ Insert into appropriate buckets
         G:1.3 → Bucket 2 [1.0-1.5mm]
         H:1.6 → Bucket 3 [1.5-2.0mm]
         I:1.4 → Bucket 2 [1.0-1.5mm]
         J:1.8 → Bucket 3 [1.5-2.0mm]

Bucket 2 [1.0-1.5mm]:  Process all nodes with cost 1.0 - 1.5
┌─────────────────────────────────────────────────┐
│ Bucket 2: [D:1.2, G:1.3, I:1.4]                │
│ Expand in parallel...                           │
└─────────────────────────────────────────────────┘

CORRECT: Nodes always processed in cost order!
         Bucket 0 → Bucket 1 → Bucket 2 → ...
```

---

## Detailed Algorithm Flow

### Step 1: Initialization

```python
# Create bucket structure (K ROIs × max_buckets × frontier_words)
max_buckets = ceil(1000.0 / delta)  # For delta=0.5, max_buckets=2000
buckets = cp.zeros((K, max_buckets, frontier_words), dtype=cp.uint32)

# Place source nodes in bucket 0 (distance 0)
for roi_idx in range(K):
    src = data['sources'][roi_idx]
    word_idx = src // 32
    bit_pos = src % 32
    buckets[roi_idx, 0, word_idx] = (1 << bit_pos)
```

**Visual:**
```
ROI 0: Bucket 0: [src=42] ✓   Bucket 1: []   Bucket 2: []   ...
ROI 1: Bucket 0: [src=17] ✓   Bucket 1: []   Bucket 2: []   ...
ROI 2: Bucket 0: [src=91] ✓   Bucket 1: []   Bucket 2: []   ...
...
```

### Step 2: Find First Non-Empty Bucket

```python
current_bucket = 0
while current_bucket < max_buckets:
    if buckets[:, current_bucket, :] has nodes:
        break  # Found non-empty bucket
    current_bucket += 1
```

**Visual:**
```
Scan buckets: 0 ✓ (has nodes) → Process bucket 0
```

### Step 3: Extract and Clear Bucket

```python
# Extract nodes in current bucket
frontier = buckets[:, current_bucket, :].copy()

# Clear bucket (nodes may be reinserted if improved)
buckets[:, current_bucket, :] = 0
```

**Visual:**
```
BEFORE:
Bucket 0: [A:0.0, B:0.4] ✓

EXTRACT:
frontier = [A:0.0, B:0.4]

AFTER:
Bucket 0: [] (cleared)
```

### Step 4: Expand Nodes (Edge Relaxation)

```python
# Process all nodes in frontier (current bucket)
nodes_expanded = _delta_relax_bucket(data, K, frontier, buckets, current_bucket, delta)

# Inside _delta_relax_bucket:
for each node in frontier:
    for each neighbor of node:
        new_dist = dist[node] + edge_cost
        if new_dist < dist[neighbor]:
            dist[neighbor] = new_dist
            parent[neighbor] = node
            # Assign neighbor to appropriate bucket
            bucket_id = floor(new_dist / delta)
            buckets[roi_idx, bucket_id, neighbor] = 1
```

**Visual:**
```
Node A:0.0 ──[0.8]──> Node C
              new_dist = 0.0 + 0.8 = 0.8
              bucket_id = floor(0.8 / 0.5) = 1
              Insert C into Bucket 1 ✓

Node A:0.0 ──[1.2]──> Node D
              new_dist = 0.0 + 1.2 = 1.2
              bucket_id = floor(1.2 / 0.5) = 2
              Insert D into Bucket 2 ✓

Node B:0.4 ──[0.5]──> Node E
              new_dist = 0.4 + 0.5 = 0.9
              bucket_id = floor(0.9 / 0.5) = 1
              Insert E into Bucket 1 ✓
```

### Step 5: Bucket State After Expansion

```
Bucket 0: [] (empty - processed)
Bucket 1: [C:0.8, E:0.9] ← New nodes discovered
Bucket 2: [D:1.2] ← New nodes discovered
Bucket 3: []
...
```

### Step 6: Loop Until Bucket Empty

```python
while bucket[current_bucket] not empty:
    # Extract, clear, expand (repeat steps 3-4)
    frontier = buckets[:, current_bucket, :].copy()
    buckets[:, current_bucket, :] = 0
    nodes_expanded = _delta_relax_bucket(...)
```

**Key Property:** Nodes can be reinserted into the SAME bucket if a better path is found!

**Example:**
```
Iteration 1:
  Bucket 1: [C:0.8, E:0.9]
  Expand C:0.8 → discovers F:1.0
  Insert F into Bucket 2

Iteration 2:
  Bucket 1: [] (empty now)
  Advance to Bucket 2

Bucket 2: [D:1.2, F:1.0]
  Expand D:1.2 → discovers alternate path to F: 1.1
  REINSERT F into Bucket 2 (improved distance!)

Continue until Bucket 2 empty...
```

### Step 7: Advance to Next Bucket

```python
if bucket[current_bucket] empty:
    current_bucket += 1
    # Process next bucket (step 2)
```

### Step 8: Termination

```python
# Check if all sinks reached
sink_dists = data['dist'][cp.arange(K), data['sinks']]
sinks_reached = int(cp.sum(sink_dists < 1e9))

if sinks_reached == K:
    break  # All paths found!

if current_bucket >= max_buckets:
    break  # All buckets processed
```

---

## Bucket Assignment Examples

### Example 1: Grid-like Routing (delta = 0.5mm)

```
Edge costs: 0.4mm (grid pitch)

Source A:0.0
  ├─[0.4]→ B:0.4   → Bucket 0 [0.0-0.5)
  ├─[0.4]→ C:0.4   → Bucket 0 [0.0-0.5)
  └─[0.4]→ D:0.4   → Bucket 0 [0.0-0.5)

Node B:0.4
  ├─[0.4]→ E:0.8   → Bucket 1 [0.5-1.0)
  ├─[0.4]→ F:0.8   → Bucket 1 [0.5-1.0)
  └─[0.4]→ G:0.8   → Bucket 1 [0.5-1.0)

Node E:0.8
  ├─[0.4]→ H:1.2   → Bucket 2 [1.0-1.5)
  ├─[0.4]→ I:1.2   → Bucket 2 [1.0-1.5)
  └─[0.4]→ J:1.2   → Bucket 2 [1.0-1.5)

Bucket progression: 0 → 1 → 2 → 3 → ...
Each bucket contains ~2-3 hops of neighbors
```

### Example 2: Via-Heavy Routing (delta = 0.5mm)

```
Edge costs: 0.4mm (grid) + 3.0mm (via)

Source A:0.0
  ├─[0.4]→ B:0.4    → Bucket 0 [0.0-0.5)
  └─[3.0]→ C:3.0    → Bucket 6 [3.0-3.5)  (via!)

Node B:0.4
  ├─[0.4]→ D:0.8    → Bucket 1 [0.5-1.0)
  └─[3.0]→ E:3.4    → Bucket 6 [3.0-3.5)  (via!)

Node D:0.8
  ├─[0.4]→ F:1.2    → Bucket 2 [1.0-1.5)
  └─[3.0]→ G:3.8    → Bucket 7 [3.5-4.0)  (via!)

Bucket progression: 0 → 1 → 2 → ... → 6 → 7 → ...
Horizontal routing (cheap) expands first
Via-heavy paths (expensive) deferred to later buckets
```

---

## Multi-ROI Parallel Execution

Delta-stepping supports batched execution across K ROIs:

```
┌─────────────────────────────────────────────────┐
│ ROI 0: Bucket 2: [node_5, node_12, node_34]    │
│ ROI 1: Bucket 2: [node_8, node_19]             │  } Processed
│ ROI 2: Bucket 2: [node_42, node_73, node_91]   │  } in parallel
│ ...                                              │  } (GPU kernel)
│ ROI 31: Bucket 2: [node_17, node_28]           │
└─────────────────────────────────────────────────┘
         │
         ▼ Single GPU kernel launch
     All ROIs' bucket 2 nodes expanded together
```

**Benefit:** GPU processes 1000s of nodes across 32 ROIs in single kernel launch!

---

## Performance Characteristics

### Bucket Count Impact

```
Delta = 0.4mm (small):
  ├─ Buckets: 2500 (many)
  ├─ Precision: High (fine granularity)
  ├─ Overhead: Higher (more bucket scans)
  └─ Use: When exact optimal paths needed

Delta = 0.5mm (balanced): ← DEFAULT
  ├─ Buckets: 2000 (moderate)
  ├─ Precision: Good (1.25× grid pitch)
  ├─ Overhead: Reasonable
  └─ Use: General purpose

Delta = 0.8mm (large):
  ├─ Buckets: 1250 (fewer)
  ├─ Precision: Medium (2× grid pitch)
  ├─ Overhead: Lower (fewer scans)
  └─ Use: When speed prioritized

Delta = 1.6mm (very large):
  ├─ Buckets: 625 (very few)
  ├─ Precision: Low (degenerates to Dijkstra)
  ├─ Overhead: Minimal
  └─ Use: Coarse approximation
```

### Memory Usage

```
Bucket memory = K × max_buckets × frontier_words × 4 bytes

Example (typical ROI):
  K = 32 ROIs
  max_buckets = 2000 (delta=0.5mm)
  frontier_words = 3,126 (100K nodes)
  memory = 32 × 2000 × 3,126 × 4 = 800 MB ✓

Example (full graph - worst case):
  K = 32 ROIs
  max_buckets = 2000
  frontier_words = 131,250 (4.2M nodes)
  memory = 32 × 2000 × 131,250 × 4 = 33.6 GB ✗ (too large!)

Solution: Use ROIs (not full graph) to keep memory manageable
```

---

## Comparison Table

| Aspect | BFS Wavefront | Δ-Stepping |
|--------|--------------|-----------|
| **Cost Ordering** | ❌ Ignores | ✅ Respects |
| **Path Quality** | Suboptimal | ✅ Optimal |
| **Correctness** | ❌ Violates Dijkstra | ✅ Correct |
| **Memory** | Low (1 frontier) | Higher (K × buckets) |
| **Overhead** | Minimal | Bucket management |
| **Parallelism** | High (all nodes) | High (per bucket) |
| **Use Case** | Debug/baseline | Production |

---

## Visual Summary

```
BFS WAVEFRONT (OLD):
┌──────────┐
│ Frontier │ → Process ALL nodes → Done
└──────────┘
           ↓
      [Mixed costs: 0, 5, 10, 15, 20, 25]
      ❌ WRONG ORDER!


Δ-STEPPING (NEW):
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Bucket 0 │ →   │ Bucket 1 │ →   │ Bucket 2 │ → ...
│ [0-0.5)  │     │ [0.5-1.0)│     │ [1.0-1.5)│
└──────────┘     └──────────┘     └──────────┘
     ↓                ↓                 ↓
  [0.0, 0.4]      [0.8, 0.9]       [1.2, 1.4]
  ✅ CORRECT ORDER!
```

---

## Key Takeaways

1. **Δ-stepping processes nodes in cost order** (buckets 0, 1, 2, ...)
2. **Each bucket contains nodes with similar costs** (range = delta)
3. **Buckets drained before advancing** (ensures correctness)
4. **Nodes can be reinserted** (if better path found)
5. **Delta parameter trades precision vs performance** (0.5mm is good default)
6. **Multi-ROI batching maintained** (K ROIs processed in parallel)
7. **Memory scales with buckets** (use ROIs to keep manageable)

**Result:** Correct shortest-path guarantees + optimal path quality!
