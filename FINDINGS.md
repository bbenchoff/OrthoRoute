# Hardening audit findings — branch `hardening/audit-2026-07`

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

_(pending)_

## Phase 5 — Hygiene: constants, pins, cost budget

_(pending)_

## Phase 6 — Per-net phase instrumentation

_(pending)_

## Phase 7 — Strict invariant mode

_(pending)_

## Phase 8 — God-class decomposition

_(pending)_

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
