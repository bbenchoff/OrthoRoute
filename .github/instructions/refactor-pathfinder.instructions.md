---
applyTo: "**/unified_pathfinder.py"
---

# Refactoring Rules for unified_pathfinder.py

This file is a **3,900+ line monolith**. It works. Treat it with care.

## Prime Directive

> **Never restructure the whole file in one pass.** One extraction at a time, one test before each extraction.

## Safe Workflow

1. **Identify one cohesive chunk** (≤200 lines, single responsibility).
2. **Write a test** that exercises the current behavior of that chunk.
3. **Run the test** — confirm it passes against the original code.
4. **Extract** the chunk into a new class/module under `orthoroute/algorithms/manhattan/pathfinder/`.
5. **Replace** the original with a call to the new class.
6. **Run the test again** — must still pass.
7. Stop. Do not continue to the next chunk in the same session.

## What NOT to Do

- Do not rename methods across the whole file to fix style.
- Do not add type annotations file-wide.
- Do not reorder methods for readability.
- Do not change any algorithm logic while refactoring — structure only.

## Good First Extractions (pre-identified)

| Chunk | Approximate lines | Suggested class name |
|-------|-------------------|----------------------|
| Cost matrix initialization | ~150 | `CostMatrixBuilder` |
| Keepout obstacle marking | ~120 | `KeepoutObstacleApplier` |
| Pad-to-lattice mapping | ~180 | `PadLatticeMapper` |
| Batch net scheduling | ~200 | `NetBatchScheduler` |

## After Extraction

Place new files in `orthoroute/algorithms/manhattan/pathfinder/` (directory already exists). Update `__init__.py` exports. Do not change the public API of `UnifiedPathFinder` itself.
