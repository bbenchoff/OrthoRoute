# Manhattan Router — Design Document

This document explains what the current algorithm does and how the pieces fit:

Where relevant, I’ve folded in the recent changes you’ve been testing (e.g., the **full-graph CSR + ROI bitmap** approach and the **FIX-7** bitmap checks in CUDA) and the Manhattan–grid invariants (H/V layer discipline, 0.4 mm pitch, blind/buried via constraints).

---

## 1) High-level flow (what a full run does)

1. **Ingest the board + lattice setup**

   * Establish a rectilinear lattice at 0.4 mm pitch.
   * Assign **layer discipline**: e.g., In1.Cu = horizontal, In2.Cu = vertical, In3.Cu = horizontal, … (H/V/H/V).
   * Define legal blind/buried via pairs (which Z transitions are allowed) and their cost.

2. **Build the global routing graph (once)**

   * Nodes are grid points `(x, y, z)`.

   * Edges exist only for **legal Manhattan moves**: ±X on H-layers, ±Y on V-layers, and ±Z via edges between allowed layer pairs.

   * Weights = base cost + via cost (for Z edges) + design-rule penalties (spacing/keepouts) + **negotiated congestion penalties** (PathFinder).

   > Stored in GPU-friendly CSR: `indptr`, `indices`, `weights` for the **full graph**.

3. **Iterative routing (PathFinder negotiation)**

   * Run for `N_iter` (e.g., 10) negotiations.
   * Each negotiation iteration:

     1. Batch a subset of **unrouted or ripped** nets (e.g., K ≈ 152).
     2. For each net, build a **Region of Interest (ROI)** that’s just big enough to plausibly connect source→sink while respecting layer discipline (the “symmetric L-corridor” you see in logs).
     3. **Route K nets in parallel on the GPU** using the global CSR **filtered by ROI bitmaps** (FIX-7).
     4. Turn the GPU parent arrays into **grid paths**, then into **tracks + vias**. Enforce Manhattan adjacency; compress collinear runs; drop illegal diagonals.
     5. Update congestion costs for **overused edges** (if any). Nets that failed or used overfull edges will be reattempted in the next iteration with higher penalties.

4. **Finalize**

   * When all iterations finish, you’ve got: near-100% routed nets, **zero over-use**, and geometry that’s locked to the Manhattan grid.

---

## 2) Medium-level breakdown (the subsystems)

### A. Lattice & rule encoding

* **Grid definition:** pitch, board extents per layer.
* **Layer roles:** H or V movement per copper layer; **vias** limited to legal Z pairs; via cost table.
* **Obstacles & keepouts:** projected into the grid as blocked nodes or high-cost edges.

**Invariants**

* On any single layer, valid adjacency is 4-connected but **axis-aligned** only (no diagonals).
* Z edges exist **only** where a via is legal.

---

### B. Global graph (CSR)

* **Nodes:** 0..`N_global-1`, encode `(x,y,z)` (and decode math is pure integer).
* **Edges:** CSR arrays on GPU once per run:

  * `indptr[u]..indptr[u+1]` enumerate neighbors.
  * `indices[e]` neighbor node id.
  * `weights[e]` base + via + congestion penalties.

**Why global CSR?**

* Avoids per-net CSR extraction (expensive).
* With **ROI bitmap filtering in the kernels**, we can keep the big CSR and still confine relaxations to each net’s corridor.

---

### C. ROI generator (per net)

* **Input:** net src/dst `(x_s,y_s,z_s)` and `(x_t,y_t,z_t)`, board obstacles, current congestion.
* **Output:** an **irregular set of node ids** that define a **symmetric L-corridor** spanning the plausible Manhattan routes between the pins (across allowed Z).

  * Also produce the ROI **bounding box** `[minx..maxx]×[miny..maxy]×[minz..maxz]` (cheap checks) and a **bitmask** (strict checks).
* **Budgeting:** capps at ~500–600k nodes per ROI to avoid runaway cost while keeping connectivity.

**What the logs show**

* Lines like `Symmetric L-corridor ROI: 606,210 nodes (both L-paths included), Z 0-4`.
* Then `Using full-graph CSR + GPU bitmap (606210/2474712 nodes, 77335 words)`: that’s the ROI size, the global graph size, and the bitmap footprint in 32-bit words.

---

### D. Batch packer

* Select up to **K** nets (e.g., 152).
* For each:

  * Pack `src`, `dst` (global ids), the **shared global CSR** pointers, the **ROI bitmap**, and the **ROI bbox**.
* Allocate/point to batched device buffers for **dist**, **parent**, **frontiers**, etc. using **stamp counters** to avoid clearing.

---

### E. GPU routing (per batch)

Two cooperating CUDA kernels operate per iteration step:

1. **Full-scan / initialization (rare):**

   * Used to seed the first frontier or recover from very sparse frontiers.
   * Writes initial distances, parents, and frontier bits.

2. **Wavefront expansion (active-list):**

   * Iterates until **sink reached** or **frontier empty**.
   * For every active node:

     * Iterate outgoing edges via global CSR.
     * **FIX-7: Check ROI bitmap before relaxing**. If the neighbor isn’t in this net’s ROI, skip it even if it’s inside the bbox.
     * Compute tentative `g_new = g[u] + w(u→v)`.
     * Apply optional **A*** term `h(v)` using **on-the-fly coordinate decoding** (no global loads).
     * If `g_new` improves, atomically update `dist[v]`, set `parent[v] = u`, and mark `v` in `new_frontier`.

**Crucial details**

* **Bit-packed frontiers** (uint32) to keep memory traffic low.
* **Per-net strides** into big pools: `(K, N_max)`; use **generation stamps** to avoid memsets.
* **Parent writes are guarded** by distance improvement; races are benign because distance is the gate.

---

### F. Path extraction & geometry emission

* After kernels report success:

  * Backtrack **parent** from `dst` to `src` to recover a **global id path** (already global when using the full CSR).
  * **Host-side validation**: for each step, ensure **Manhattan adjacency** (`|Δx|+|Δy|+|Δz| == 1`), and that layer moves match H/V discipline.
  * **Run-length compress** axis-aligned chains to single segments; insert vias on Z steps.
  * Commit tracks and update congestion usage for the next negotiation round.

---

### G. Negotiated congestion (PathFinder)

* After each batch and iteration:

  * Compute **overuse** per edge; bump **present factor** (`pres_fac`) and/or **history cost** where needed.
  * Rip up any nets using overfull edges; re-route them with elevated costs in the next iteration.
* Converges to **zero overuse** with high success rate (your 99%+ result).

---

## 3) Close to the metal (data & loops)

### Coordinates & ids

* **Indexing:** `id = ((z * Ny) + y) * Nx + x`
* **Decode:** `x = id % Nx`, `y = (id // Nx) % Ny`, `z = id // (Nx*Ny)`
* **Layer discipline:**

  * If `z` is H-layer: allow neighbors `±X` only.
  * If `z` is V-layer: allow neighbors `±Y` only.
  * If `(z,z')` is a legal via pair: allow `±Z` with `via_cost`.

### Global CSR (GPU-resident)

* `indptr`: `int32[N_global+1]`
* `indices`: `int32[E_global]`
* `weights`: `float32[E_global]` (includes present/history penalties and via costs)

### Per-batch arrays (device)

* `dist`: `float32[K, N_max]` (sliced view into a pool)
* `parent`: `int32[K, N_max]`
* `frontier`, `new_frontier`: `uint32[K, ceil(N_max/32)]`
* `roi_bitmap`: `uint32[K, ceil(N_global/32)]`  ← **FIX-7 critical**
* `roi_minx/maxx, roi_miny/maxy, roi_minz/maxz`: `int32[K]`

**Stamping trick**

* Each slice has a **generation counter**. Instead of zeroing, kernels read the stamp; if it mismatches the active generation, treat as “unset.”

### Kernel inner loop (essential steps)

For each active `(roi_idx, u)`:

1. Iterate `e in indptr[u]..indptr[u+1]`:
2. `v = indices[e]`
3. **Bitmap gate:**

   ```
   word = v >> 5
   bit  = v & 31
   if ((roi_bitmap[roi_idx, word] >> bit) & 1) == 0: continue
   ```
4. **Layer gate (compile-time or weight encoding):** ensure `(u→v)` respects H/V.
5. `g_new = dist[u] + weights[e]`
6. If `g_new < dist[v]`:

   * `dist[v] = g_new` (atomic min or CAS on (dist,parent))
   * `parent[v] = u`
   * Set bit for `v` in `new_frontier[roi_idx]`

When `dst` is discovered (under a cut-off) the net can early-exit.

### Geometry emission (host)

* Backtrack `parent` to list `[id_0=src, id_1, …, id_n=dst]`.
* **Assert** for each consecutive pair:

  * `Δz ∈ {0, ±1}`, `Δx, Δy ∈ {0, ±1}`, and `|Δx|+|Δy|+|Δz| == 1`.
  * If `Δz == 0`: enforce H/V per layer.
* Coalesce axis-aligned runs into single tracks with endpoints snapped to **exact 0.4 mm centers**; insert vias at `Δz` steps.

---

## 4) Performance profile & current bottlenecks

* **GPU routing**: extremely fast when the ROI is correct; you’ve seen billions of edge relaxations/sec and stable 152/152 ROI concurrency.
* **ROI creation**: dominant cost in your logs. You’re seeing many ROIs of **~0.5–0.6 M nodes**, built **sequentially**. The log cadence (~2–4 lines/sec) matches **CPU-bound ROI bitmap building**.
* **Transfers**: a per-net bitmap is ~77,335 words ≈ 302 KB. Batched K=152 → ~46 MB to hand to kernels; OK if reused, expensive if rebuilt per net per iteration.

**Low-risk speedups**

* **Keep the full-graph CSR.** It’s the right architecture.
* **Move ROI to GPU:** Build the **symmetric L-corridor** via a GPU flood fill / masked BFS that:

  * Starts from src/dst seed sets,
  * Enforces H/V and legal Z as it expands,
  * Stops on a step/area budget,
  * Emits the **bitmap directly on device**.
    (No Python loop: no `for u in roi_nodes_cpu:` bit-sets.)
* **Batch ROI generation:** produce ROI bitmaps for **K nets at once** with separate bit planes `(K, ceil(N_global/32))` and shared kernels.
* **Cache ROI bitmaps across a PathFinder iteration.** Only rebuild when pins move or cost gates change ROI shape materially (rare).
* **Compact bitmaps**: if PCIe bandwidth is tight, RLE or lightweight LZ4 on host→device, then decompress on GPU (fast kernel) before routing.

---

## 5) Correctness guarantees (Manhattan ideal)

* **Graph-level guarantees**

  * On-layer edges are axis-aligned only; no diagonals exist in the edge set.
  * Z moves exist only for **legal via pairs**; via cost penalizes unnecessary layer swaps.
* **Kernel-level guarantees**

  * **FIX-7 ROI bitmap check** forbids exploration outside the ROI mask (prevents parent corruption and diagonal geometry artifacts).
  * Parent writes occur **only** on strict distance improvement; invalid parents can’t form because there are no invalid edges after bitmap gating.
* **Host-level guarantees**

  * Path decoding enforces `|Δx|+|Δy|+|Δz| == 1`.
  * H/V discipline rechecked when emitting geometry.
  * Any violation is logged and rejected before track emission.

---

## 6) Testing & instrumentation

* **Per-batch counters:** Paths found `k/K`, kernel ms, edges/sec, frontier densities.
* **ROI stats:** size in nodes, bbox, bitmap words, build time (and whether it came from cache).
* **Adjacency assertions:** fatal on any non-Manhattan step.
* **Layer barcodes:** quick layer-only visual: In1/In3 segments must be horizontal; In2/In4 vertical; vias at legal Z.

---

## 7) Summary of “must-haves” for the Manhattan ideal

1. **Keep**: Global CSR + **mandatory** ROI-bitmap check in kernels (FIX-7).
2. **Enforce**: H/V layer movements and legal via pairs in both graph construction and kernel guards.
3. **Validate**: Host-side adjacency assertions before geometry emission.
4. **Accelerate**: ROI construction **on GPU** and **batch-wise**, cache across an iteration.
5. **Negotiate**: Standard PathFinder present/history costs; rip-up only what’s necessary.

That gives you the trifecta: rectilinear geometry, high success rate, and wall-clock times dominated by the GPU instead of Python. If you want, I can turn the ROI-on-GPU bit into a concrete mini-spec (kernel signatures + memory layout) next so your team can wire it in without guesswork.
