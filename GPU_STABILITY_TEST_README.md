# GPU Stability Test - Fast GPU Routing Test

## Overview

This test suite verifies the stability and correctness of GPU routing by running 50+ iterations of focused GPU pathfinding tests. Unlike the full 30-40 minute test, this fast test completes in 3-5 minutes per iteration by routing only the large nets that trigger GPU acceleration.

## Prerequisites

**CRITICAL: You must have KiCad running with a board that has nets to route!**

The test loads the board directly from KiCad using IPC, so:

1. **Open KiCad**
2. **Load a board with nets** (e.g., the MainController board or any board with 200+ nets)
3. **Keep KiCad running** while the tests execute

The test detected your board at: `C:\Users\Benchoff\Documents\GitHub\ThinkinMachine\MainController\MainController.kicad_pcb`

## Files Created

- `test_gpu_fast.py` - Fast GPU routing test (routes top 200 nets)
- `run_stability_tests.sh` - Bash script to run 50 iterations
- `test_gpu_fast.log` - Accumulated log of all test runs (append mode)
- `stability_run.log` - High-level summary of pass/fail for each run

## Running the Tests

### Step 1: Verify Prerequisites

Make sure KiCad is running with a board that has nets:

```bash
python test_gpu_fast.py
```

You should see:
- "Board loaded: XXXX nets total" (where XXXX > 0)
- "Testing with 200 large nets (will generate ~152 GPU ROIs)"
- "TEST COMPLETE: XXX/200 paths found (XX.X%)"
- "PASS: GPU routing working correctly"

If you see "Board has no nets to route", load a different board in KiCad.

### Step 2: Run Stability Tests

```bash
chmod +x run_stability_tests.sh
./run_stability_tests.sh
```

The script will:
- Run up to 50 iterations
- Stop early if more than 5 failures occur
- Print progress after each run
- Calculate running success rate
- Timeout individual tests after 5 minutes

Expected runtime: **2-4 hours** (50 runs × 3-5 minutes per run)

### Step 3: Monitor Progress

While tests are running, monitor in another terminal:

```bash
# Watch overall progress
tail -f stability_run.log

# Watch detailed GPU behavior
tail -f test_gpu_fast.log | grep -E "ROIs active|expanded=|CUDA|TEST COMPLETE"
```

## What the Test Checks

The fast GPU test:

1. ✓ Loads board from KiCad (~2s)
2. ✓ Initializes GPU pathfinder with full lattice (~5s)
3. ✓ Sorts nets by size and selects top 200
4. ✓ Routes these nets (generates ~152 GPU ROIs)
5. ✓ Verifies >80% success rate
6. ✓ Completes in 3-5 minutes (vs 30-40 minutes for full test)

This exercises the **EXACT SAME GPU CODE PATH** as the full test, but:
- Skips creating 8,000+ CPU ROIs for small nets
- Only creates the ~152 GPU-eligible ROIs
- Much faster iteration for stability testing

## Success Criteria

For production readiness, we expect:

- **50/50 tests pass** (100% success rate)
- **Every test shows `152/152 ROIs active`** (no ROI collapse)
- **No CUDA errors**
- **Success rate >90% for each run** (most nets route successfully)
- **Consistent performance** (~3-5 minutes per run)

## Analyzing Results

After tests complete (or if you stop early), analyze:

### 1. Count Runs and Successes

```bash
grep "TEST COMPLETE" test_gpu_fast.log | wc -l  # Total runs
grep "PASS: GPU routing" test_gpu_fast.log | wc -l  # Successful runs
```

### 2. Verify No ROI Collapse

```bash
grep "152/152 ROIs active" test_gpu_fast.log | head -20
```

Every test should show `152/152 ROIs active`. If you see `1/152` or similar, the bug has returned.

### 3. Check for CUDA Errors

```bash
grep -i "cuda.*error" test_gpu_fast.log
```

Should return **zero results**. Any CUDA errors indicate memory corruption or GPU issues.

### 4. Analyze Success Rates

```bash
grep "TEST COMPLETE" test_gpu_fast.log
```

Look for lines like: `TEST COMPLETE: 184/200 paths found (92.0%)`

Success rates should be >90%. Lower rates suggest routing quality issues (but not necessarily GPU bugs).

### 5. Check Average Test Time

```bash
grep "Total time:" test_gpu_fast.log
```

Should average 3-5 minutes. Significantly longer times suggest performance regressions.

## Interpreting Results

### ✓ PRODUCTION READY

- 50/50 passes (or 45+/50 with investigation of failures)
- 152/152 ROIs active in every run
- No CUDA errors
- Success rates >90%
- Stable performance

### ⚠ NEEDS INVESTIGATION

- 40-45/50 passes
- Occasional low success rates (<80%)
- Performance variance (some runs 10+ minutes)
- No CUDA errors, but quality issues

### ✗ CRITICAL BUGS

- <40/50 passes
- ROI collapse (1/152 or similar)
- CUDA errors present
- Crashes or hangs
- Success rates <50%

## Troubleshooting

### "Board has no nets to route"

**Solution:** Load a board with nets in KiCad. The MainController board should work.

### "Failed to load board from KiCad"

**Solution:** Make sure KiCad is running and has a board open.

### Test hangs or takes >10 minutes

**Possible causes:**
- Board is too large (>10,000 nets)
- GPU memory issue
- Routing convergence problem

**Action:** Check `test_gpu_fast.log` for the last iteration number. If stuck in iteration loop, this indicates a routing issue, not a GPU bug.

### CUDA errors appear

**Critical!** This indicates:
- Memory corruption
- Array bounds violation
- GPU resource exhaustion

**Action:** Stop tests immediately and investigate. Check the error message in the log.

## Quick Start (TL;DR)

```bash
# 1. Open KiCad with a board that has nets
# 2. Run one test to verify:
python test_gpu_fast.py

# 3. If it passes, run full stability suite:
./run_stability_tests.sh

# 4. Check back in 2-4 hours
# 5. Analyze results:
grep "STABILITY TEST COMPLETE" stability_run.log
grep "Success rate:" stability_run.log
```

## Expected Output

Successful run:

```
Starting stability test run at Wed, Oct 15, 2025  3:00:00 AM
Target: 50 successful GPU routing tests
============================================

----------------------------------------
Run #1 of 50 - Wed, Oct 15, 2025  3:00:00 AM
----------------------------------------
✓ Run #1: PASS (Total: 1 pass, 0 fail)
Current success rate: 100.0%

...

============================================
STABILITY TEST COMPLETE
============================================
Total runs: 50
Passed: 50
Failed: 0
Success rate: 100.0%
Completed at: Wed, Oct 15, 2025  5:30:00 AM
```

## Contact

If tests fail or you see unexpected behavior, provide:

1. Final success rate from `stability_run.log`
2. Any CUDA errors from `grep -i cuda test_gpu_fast.log`
3. ROI status from `grep "ROIs active" test_gpu_fast.log | head -20`
4. Last 50 lines of `test_gpu_fast.log`
