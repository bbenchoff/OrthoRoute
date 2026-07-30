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

_(pending)_

## Phase 2 — Exception discipline in the GPU fast path

_(pending)_

## Phase 3 — Documentation drift

_(pending)_

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

- Phase 4 note (verified before starting): the believed-dead methods reference each other —
  `find_path_single` → `find_path_batch`, `find_path_multisource_multisink_gpu` →
  `find_paths_on_rois`, `find_paths_bidirectional_batch` → `find_path_bidirectional`.
  Deletions proceed callers-before-callees so each grep check stays meaningful.
- A Metal/MLX backend exists (`orthoroute/algorithms/manhattan/pathfinder/metal_dijkstra.py`,
  `mlx` on Darwin in requirements). The audit task doesn't mention it. It is left untouched;
  changes were checked to not import or alter it.
