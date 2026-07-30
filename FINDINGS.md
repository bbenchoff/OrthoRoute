# Hardening audit findings — branch `hardening/audit-2026-07`

## Summary

All phases complete (Phase 8: 3 of 4 extractions; the fourth skipped deliberately).

| Metric | Before | After |
|---|---|---|
| Unit+integration suite | 342 passed / 9 skipped | **345 passed / 9 skipped** (+3 new strict-mode tests) |
| Engine smoke suite | 80 passed | **80 passed** |
| Removed tests | — | **none** (no test referenced deleted code) |
| `cuda_dijkstra.py` | 5,928 lines | 3,922 |
| `unified_pathfinder.py` | 11,367 lines | ~9,220 (three collaborator modules extracted) |
| `persistent_kernel.py` | 702 lines | ~500 (dead kernel variant removed; occupancy sizing added) |

New modules: `pathfinder/cuda_common.py` (shared device preamble),
`shared/profiling.py` (ORTHO_PROFILE instrumentation), `manhattan/geometry_emitter.py`,
`manhattan/via_accounting.py`, `manhattan/hotset_policy.py`.
New env vars: `ORTHO_PROFILE=1`, `ORTHO_STRICT=1`. No routing-semantics changes anywhere.

Implementation log for the July 2026 audit hardening task. One section per phase.
Ground rules observed: no routing-semantics changes, tests are the contract, one commit
per phase.

Environment: macOS (Darwin), Python 3.12 in an isolated venv with `pytest numpy scipy psutil`
only. No CUDA hardware — all GPU-path changes are read-verified only, per the task's ground
rule 1. The Darwin requirements marker installs `mlx`, not `cupy`, so the plan's note that
"cupy will fail to build" did not apply; cupy is simply skipped.

## Phase 0 — Baseline

- `python -m pytest tests/ -q --ignore=tests/regression --ignore=tests/test_engine_smoke.py`
  → **342 passed, 9 skipped**
- `python -m pytest tests/test_engine_smoke.py -q` → **80 passed**

Exactly matches the expected baseline in the task description. The suites are fully green
with only `pytest numpy scipy psutil` installed (no PyQt6, no mlx, no cupy).

Pre-existing working-tree change carried into this branch: one `.gitignore` line adding
`/notes/` (gitignored session-documentation folder, requested by the maintainer).

## Phase 1 — Cooperative launch occupancy query

Replaced the hardcoded `num_blocks = 80` in `launch_persistent_kernel()` with
`_cooperative_grid_size(kernel, threads_per_block)`:

- SM count from `cp.cuda.Device().attributes['MultiProcessorCount']`.
- Max resident blocks/SM for this kernel from
  `cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.kernel.ptr, 256, 0)`
  (the compiled `Function` behind a `RawKernel` is `kernel.kernel`; its `.ptr` fetch is
  guarded by `AttributeError`).
- Grid = `sm_count * blocks_per_sm`, clamped to [1, 512].
- Any query failure → conservative fallback of 32 blocks with a one-line WARNING naming
  the failed query; chosen config logged once at DEBUG.
- Result cached in a module dict keyed by (function pointer, block size) — computed once
  per process.

CuPy import remains lazy/guarded (module-level `try/except ImportError` was already
present); code is unreachable without CuPy. Only one launch site exists
(`cuda_dijkstra.py:5754`), unchanged. Tests: 342/9 + 80, green.

## Phase 2 — Exception discipline in the GPU fast path

In `_route_all`'s handler around `find_path_fullgraph_gpu_seeds`:

- Added module-level `_is_cuda_infrastructure_error()` — true when the exception type's
  module starts with `cupy` (covers `cupy.cuda.runtime.CUDARuntimeError`,
  `cupy.cuda.memory.OutOfMemoryError`, and `cupy_backends.*` driver errors). No new
  top-level cupy import.
- CUDA infrastructure errors: first occurrence per process logs an actionable WARNING
  (names the exception, notes the ~10x CPU fallback and the cooperative-launch/SM-count
  suspicion); repeats log at DEBUG with a running count. Per-run counter
  `self._gpu_fastpath_cuda_failures` is reported in the existing `[GPU-STATS]`
  end-of-run summary.
- Non-CUDA exceptions: unchanged WARNING, plus the full traceback now logged at DEBUG
  (`exc_info=True`) so real bugs aren't invisible.
- Fallback behavior and the expected no-path `None` flow are untouched.

Tests: 342/9 + 80, green.

## Phase 3 — Documentation drift

Documentation-only changes (no code):

- Core-loop step (d) and negotiation STEP 3 now name the full-graph GPU supersource
  label-correcting search as the primary path, ROI heap Dijkstra as the CPU fallback.
- Layer counts: header now states 32-layer support and the 32-layer flagship board up
  front, marks the remaining 18-layer material as examples, generalized "all 18 layers"
  and "B.Cu (L17)".
- RESULTS block replaced with pointers to `docs/optimization/` and
  `tests/regression/golden_metrics.json` (both verified to exist).
- "Dijkstra" prose for the GPU solver corrected to frontier/queue label-correcting
  (Bellman-Ford family) in the module docstring, in the `cuda_dijkstra.py` header
  (with an explicit "module name is historical" note — no file/class renames), and in
  the README's two "parallel Dijkstra" mentions.
- Beyond the plan's list, two sections that directly contradicted the corrected claims
  were also fixed (still prose-only): "GPU SUPPORT (currently disabled)" — it is the
  primary runtime path — and two "(TO BE IMPLEMENTED)" tags on portal machinery that
  has long been implemented (the listed-but-nonexistent methods `_route_with_portals`,
  `_emit_portal_geometry`, `_retarget_failed_portals`,
  `_gpu_roi_near_far_sssp_with_metrics` were removed from the method list).

Tests: 342/9 + 80, green.

## Phase 4 — Solver graveyard purge

`cuda_dijkstra.py`: 5,928 → 3,922 lines. `persistent_kernel.py`: 702 → 501 (net of the
Phase 1 addition). Every deletion was preceded by a repo-wide grep (tests, benchmarks,
scripts, main.py, and getattr-style dynamic dispatch checked); the unit suite ran green
after each cut.

**Important correction to the audit's dead list** — `find_paths_on_rois` is NOT dead:
the live entry point `find_path_roi_gpu` wraps its single ROI into a batch and calls it.
That keeps the whole near-far batch pipeline alive: `_prepare_batch`, `_normalize_batch`,
`_run_near_far`, the wavefront/compaction/persistent/delta-stepping helpers,
`_reconstruct_paths`, and the `K_pool` pool arrays (`dist_val_pool`, `dist_stamp_pool`,
`near_bits_pool`, …). All kept.

Deleted (no reference anywhere outside the dead cluster itself; internal call chains
removed callers-before-callees):

- `find_path_single` → `find_path_batch` (only caller of it)
- `route_batch_persistent`
- `find_path_multisource_multisink_gpu` → `_prepare_batch_multisource`,
  `_run_near_far_multisink` → `_relax_near_bucket_gpu`, `_advance_threshold`,
  `_split_near_far_buckets` (multisink loop was their only caller)
- `find_paths_bidirectional_batch` → `find_path_bidirectional` → `_transpose_csr`,
  `_unpack_frontier`, `_expand_frontier_single`
- Bonus dead helpers found during verification (not in the audit list, provably
  uncalled): `_slice_per_roi`, `_relax_edges_parallel`
- Dead kernels whose last user was deleted (or never existed): `relax_kernel`,
  `procedural_neighbor_kernel`, `persistent_kernel_stamped` (~535 lines),
  `accountant_kernel`; in `persistent_kernel.py`, the never-compiled
  `PERSISTENT_SSSP_KERNEL_CODE` bit-packed variant (~240 lines; only the QUEUE variant
  is used by `create_persistent_kernel`)
- Orphaned instance attribute `_persistent_kernel_version` (no reader, getattr included)

Kept and worth noting:
- `_fallback_cpu_dijkstra` (now cuda_dijkstra.py:3247) also has zero callers, but it was
  not on the audit's list and is the only CPU-correctness fallback living inside the GPU
  solver class, so it was left in place. Candidate for a future cut — or for re-wiring.
- `USE_DELTA_STEPPING`/`USE_PERSISTENT_KERNEL` paths inside `_run_near_far` are disabled
  by config flags but reachable; untouched.

Removed tests: none — no test referenced any deleted symbol (test counts unchanged).
Kept — referenced by tooling: none (benchmarks/ and scripts/ reference no dead entry
point).

Preamble dedup: new `pathfinder/cuda_common.py` exports `DEVICE_PRELUDE`
(`atomicMinFloat`, `f2u`, `pack_key`, `atomicMinDistanceKey`). Prepended to
`wavefront_kernel`, `active_list_kernel`, `persistent_kernel` (cuda_dijkstra.py) and
`PERSISTENT_QUEUE_SSSP_KERNEL_CODE` (persistent_kernel.py); local copies removed.
Verified by token-level diff of assembled source vs. original: `wavefront_kernel` is
token-identical; the other three gain only the (previously absent, unused) helper
functions and lose nothing — the five original copies differed only in whitespace,
parameter names, and `__forceinline__` qualifiers. Note: `via_kernels.py` contains no
copies of these functions (audit said it did; it does not).

Tests after phase: 342 passed / 9 skipped + 80 passed — identical to baseline.

## Phase 5 — Hygiene: constants, pins, cost budget

1. **Backtrace status codes**: `BACKTRACE_OK/TRUNCATED/OUT_OF_RANGE/SELF_LOOP` module
   constants in `cuda_dijkstra.py`, used in `_backtrace_fullgraph_path`'s status check.
   Truncation (length hit capacity) now logs its own distinct message; out-of-range and
   self-loop parents get specific messages too. Kernel source unchanged (constants
   documented as "keep in sync with the kernel source").
2. **Dependency pins**: `requirements.txt` floor raised `cupy>=10.0.0` → `cupy>=13`
   with a note pointing at `cupy-cuda12x>=13` wheels; `setup.py` gpu extra aligned
   (`cupy>=13`). `requirements-kicad.txt` already pinned `cupy-cuda12x>=13.4,<14` — the
   authoritative KiCad-env pin was already correct; README's CUDA 12 text needed no
   change.
3. **Float32 cost budget**: comment block in `PathFinderConfig` next to the cleanup
   penalties documenting the magnitudes (base 0.4, pres_fac ≤1024, owner 25×pres_fac,
   cleanup 1e6, quantum ~0.06 at 1e6). New `warn_if_penalties_exceed_float32_budget()`
   (warning only, never raises) checks the cleanup/reservation penalties and
   owner/path-node × peak-pres_fac products against 2^23 × grid_pitch; wired into
   `PathFinderRouter.__init__` next to the existing config logging. Defaults are all
   inside the budget, so no warning fires out of the box.

Tests: 342/9 + 80, green.

## Phase 6 — Per-net phase instrumentation

New `orthoroute/shared/profiling.py`, gated on `ORTHO_PROFILE=1` (read once at import
into a module bool):

- `profile_span(label)` — context manager; when enabled it opens an NVTX range (CuPy
  resolved lazily and only when enabled; absence is harmless) and accumulates
  `perf_counter` deltas into a module dict. When disabled it returns a shared no-op
  singleton — one boolean check and no dict writes, no perf_counter, no NVTX (verified
  by test: `profile_span(x) is _NULL_SPAN`).
- `log_profile_summary()` — logs one `[PROFILE] label=1.23s ...` line (sorted by cost)
  and resets; wired into `_pathfinder_negotiation` right after the `[ITER n]` line.

Instrumented regions:
- `find_path_fullgraph_gpu_seeds` (cuda_dijkstra.py): `seed_prep` (terminal dedupe +
  transfers + node-penalty upload), `pool_reset` (pool alloc/fills/key init),
  `bitmap_setup` (frontier scatter + owner bitmap build — two spans, same label),
  `kernel` (persistent launch, or the whole multi-launch loop), `backtrace`.
- `_route_all` (unified_pathfinder.py): `clear_path`, `owner_penalty`, and — on both
  the GPU fast path and the ROI fallback commit block — `commit_path`, `via_ownership`,
  `tracking`.

Checked: both suites green with and without `ORTHO_PROFILE=1`; a live smoke test with
`ORTHO_PROFILE=1` produced `[PROFILE] kernel=0.09s seed_prep=0.02s`.

## Phase 7 — Strict invariant mode

The warn-and-continue on `verify_present_matches_canonical()` failure in
`_pathfinder_negotiation` moved into a small module function
`enforce_present_matches_canonical(accounting, iteration)` (so it is testable without
a router): default behavior identical (WARNING, continue); with `ORTHO_STRICT=1` it
raises `RuntimeError` naming the iteration. New `tests/unit/test_strict_invariant.py`
(3 tests, modeled on `tests/unit/test_edge_accountant.py`): corrupt `present` directly
→ default warns / strict raises; clean accountant passes in both modes.

Tests: **345 passed / 9 skipped** (+3 new) + 80, green.

## Phase 8 — God-class decomposition

Three of the four extractions completed, one commit each, both suites green after each.
`unified_pathfinder.py`: 11,367 → ~9,220 lines. All extractions use delegation: method
bodies moved verbatim with `self.` → `self._router.`; thin delegating methods stay on
`PathFinderRouter` so every internal and external call site is unchanged.

1. **Geometry emission** → `manhattan/geometry_emitter.py` (`GeometryEmitter`, 10
   methods, ~515 lines). `GeometryPayload` moved with it and is re-imported by
   `unified_pathfinder` for compatibility.
2. **Via accounting & barrel ownership** → `manhattan/via_accounting.py`
   (`ViaAccounting`, 15 methods, ~1,150 lines).
3. **Hotset & stagnation policy** → `manhattan/hotset_policy.py` (`HotsetPolicy`, 13
   methods incl. the `_rolling_progress_*`/`_pressure_*` pair and two staticmethods,
   ~700 lines). The HOTSET MECHANISM and BLIND/BURIED VIA docstring sections moved
   from the giant header into these modules (geometry had no dedicated section).
4. **Escape/portal planning — SKIPPED**, per the plan's own bail-out clause. The
   router carries ~40 portal/escape methods entangled with accounting, hotsets, via
   ownership, and GPU seeding, and they overlap conceptually with the existing
   1,785-line `pad_escape_planner.py`. The reconciliation is not obvious on inspection;
   attempting it would have been rewriting, not moving.

Notable mechanics discovered during extraction (why the test contract mattered):
- The engine smoke tests construct routers with `object.__new__(PathFinderRouter)` and
  even call methods unbound with duck-typed fixture objects
  (`UnifiedPathFinder._rank_stagnation_offenders(fixture, ...)`). Collaborators are
  therefore exposed as `functools.cached_property` (geometry, via accounting) or
  constructed inline in the delegator (hotset policy) rather than assigned in
  `__init__`.
- Five multi-line `getattr(self, "...")` reads (line-wrapped, so the mechanical
  `getattr(self, ` transform missed them) initially pointed at the collaborator
  instead of the router. The smoke suite caught the behavioral difference
  (`_effective_history_hotset_cap` returned 256 instead of 512; hotset composition
  changed) before commit — fixed and re-verified.

Tests after Phase 8: **345 passed / 9 skipped + 80 passed** — baseline plus the three
Phase 7 tests, nothing lost.

## Unanticipated findings

_(collected as encountered; recorded, not fixed, per the task rules)_

- **`GPUConfig.USE_PERSISTENT_KERNEL = False`** (unified_pathfinder.py ~line 578) with the
  comment "DISABLED: Hangs on cooperative kernel launch; using wavefront with atomic keys
  instead." The Phase 1 occupancy fix addresses the most likely cause of exactly that hang
  (a grid of 80 blocks that can't all be resident). Worth re-testing the persistent kernel
  on real hardware with the new grid sizing before considering it permanently dead. Not
  flipped here — that would be a behavior change.

- Phase 4 note (verified before starting): the believed-dead methods reference each other —
  `find_path_single` → `find_path_batch`, `find_path_multisource_multisink_gpu` →
  `find_paths_on_rois`, `find_paths_bidirectional_batch` → `find_path_bidirectional`.
  Deletions proceed callers-before-callees so each grep check stays meaningful.
- A Metal/MLX backend exists (`orthoroute/algorithms/manhattan/pathfinder/metal_dijkstra.py`,
  `mlx` on Darwin in requirements). The audit task doesn't mention it. It is left untouched;
  changes were checked to not import or alter it.
- The audit's Phase 4 dead list wrongly included `find_paths_on_rois` and the K_pool
  batch machinery — they are live via `find_path_roi_gpu` (details in Phase 4).
- `_fallback_cpu_dijkstra` (cuda_dijkstra.py) has zero callers but was not on the audit
  list; left in place as the only in-class CPU fallback. Future-cut candidate.
- The engine smoke tests call some router methods unbound with duck-typed fixture
  objects — any future refactor of `_rank_stagnation_offenders` /
  `_select_stagnation_victims` / `_effective_history_hotset_cap` must preserve that.
- The docstring FILE ORGANIZATION class list carried stale line numbers ("line ~380"
  etc., off by hundreds of lines); dropped the numbers and added the collaborator
  modules during Phase 8.
- `benchmarks/` and `scripts/` reference no solver entry points at all (nothing had to
  be kept for tooling).
