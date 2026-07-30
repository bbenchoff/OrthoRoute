---
name: pathfinder-extract
description: Safely extract one cohesive chunk from unified_pathfinder.py using the test-first refactor workflow
argument-hint: "Name the chunk to extract (e.g. 'CostMatrixBuilder', 'KeepoutObstacleApplier') or describe the behavior"
agent: agent
tools: [read, search, edit, execute]
---

Perform a **single, safe extraction** from [orthoroute/algorithms/manhattan/unified_pathfinder.py](../../orthoroute/algorithms/manhattan/unified_pathfinder.py) following the rules in [.github/instructions/refactor-pathfinder.instructions.md](../instructions/refactor-pathfinder.instructions.md).

## Inputs

- **chunk**: The chunk name or behavior description to extract (from the argument, or ask if not provided)

## Workflow — do NOT skip or reorder steps

1. **Identify the chunk**
   - Locate the specific lines in `unified_pathfinder.py` that form the cohesive unit.
   - Confirm it is ≤200 lines with a single clear responsibility.
   - If it exceeds 200 lines, report back and stop — do not proceed.

2. **Write the test first**
   - Create `tests/algorithms/manhattan/pathfinder/test_<chunk_snake_case>.py` (create `__init__.py` files as needed).
   - Test the *current* behavior against the original code in `unified_pathfinder.py`.
   - Use `pytest.fixture` for shared setup; `@pytest.mark.parametrize` for input tables.
   - Mock only I/O boundaries (KiCad IPC, file system, GPU VRAM allocation).

3. **Run the test — must pass green before extraction**
   ```powershell
   pytest tests/algorithms/manhattan/pathfinder/test_<chunk>.py -v
   ```
   Stop and report if any test fails before extraction.

4. **Extract the chunk**
   - Create `orthoroute/algorithms/manhattan/pathfinder/<ChunkName>.py`.
   - Move the identified lines there as a standalone class or function.
   - Update `orthoroute/algorithms/manhattan/pathfinder/__init__.py` exports.
   - In `unified_pathfinder.py` replace the original code with a call to the new class — **do not change any logic**.

5. **Run the test again — must still pass**
   ```powershell
   pytest tests/algorithms/manhattan/pathfinder/test_<chunk>.py -v
   ```

6. **Stop here.** Do not extract a second chunk in the same session. Report:
   - Lines extracted
   - New file path
   - Test file path
   - Before/after line count of `unified_pathfinder.py`

## Pre-identified chunks (from refactor-pathfinder.instructions.md)

| Chunk name | Approx. lines | Suggested class |
|------------|---------------|-----------------|
| Cost matrix initialization | ~150 | `CostMatrixBuilder` |
| Keepout obstacle marking | ~120 | `KeepoutObstacleApplier` |
| Pad-to-lattice mapping | ~180 | `PadLatticeMapper` |
| Batch net scheduling | ~200 | `NetBatchScheduler` |
