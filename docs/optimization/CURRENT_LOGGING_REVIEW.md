# Current Logging Mechanism Review

**Date**: April 3, 2026  
**File Analyzed**: `orthoroute/algorithms/manhattan/unified_pathfinder.py` (5,967 lines)

---

## Summary

OrthoRoute uses standard Python `logging` module with **287 logger calls** in `unified_pathfinder.py` alone. Produces **799 MB log files** for a 44-minute routing run due to no verbosity control — all messages always output regardless of importance.

---

## Logging Infrastructure

**Logger setup** (`unified_pathfinder.py` line 537):
```python
import logging
logger = logging.getLogger(__name__)
```

**Centralized configuration** (`shared/utils/logging_utils.py`):
- `setup_logging(settings: LoggingSettings)` — main configuration
- Dual output: Console (WARNING only) + File (DEBUG)
- Rotating file handler (prevents Windows locking issues)
- UTF-8 encoding with `errors="replace"`

---

## Logging Usage Statistics

| Level | Count | Percentage | Use |
|-------|-------|------------|-----|
| `logger.debug` | 134 | 51.3% | Algorithm internals, ROI details, cost updates, config banner, init details |
| `logger.info` | 61 | 23.4% | Remaining informational messages |
| **`logger.warning`** | **66** | **25.3%** | **Key milestones (startup, lattice, CSR, routing complete), GPU fallback, conflicts** |
| `logger.error` | 26 | 10.0% | Violations, accounting bugs, critical issues |
| **TOTAL** | **287** | | |

**Reclassification applied (April 3, 2026)**: 18 INFO→WARNING (milestones), 81 INFO→DEBUG (verbose/per-net), 1 WARNING→DEBUG (greedy progress spam). Console now shows ~66 WARNING lines per run instead of 160+ INFO lines.

---

## Existing Tag System

Good categorization tags already in use:
- `[ITER X]` — Iteration progress
- `[CONVERGENCE]` — Convergence metrics
- `[CONFIG]` — Configuration display
- `[GPU]` — GPU operations
- `[VIA-POOL]` — Via pooling system
- `[LAYER-MAP]` — Layer mapping
- `[ACCOUNTING]` — Accounting checks
- `[ROI]` — Region of interest
- `[MANHATTAN-VIOLATION]` — Constraint violations
- `[BOUNDS]` — Board bounds
- `[EXCLUDE]` — Net exclusion
- `[CLEAN]` — Final cleanup

---

## Problematic Patterns

### 1. Per-Iteration INFO Spam
**Lines**: 3691, 3795, 3913, 3973  
64 iterations × 3+ messages = 192+ INFO lines of iteration noise.

### 2. Greedy Per-Net Progress
**Line**: 4558  
512 nets ÷ 25 = 20+ `logger.warning` lines in iteration 1 alone.

### 3. ROI Debug Spam
**Lines**: 1600, 1615, 1681  
512 nets × 64 iterations × 2 messages = **65,536 debug lines**.

### 4. Config Banner
**Lines**: 2106–2117  
15+ `logger.info` lines printed every run regardless of whether config details are needed.

---

## What Was Fixed (April 3, 2026)

- ✅ Logging reclassification applied — 100 calls reclassified
- ✅ Key milestones now visible on console (WARNING)
- ✅ Verbose/per-net messages demoted to DEBUG (file only)
- ✅ Greedy per-net progress (`[ITER 1 - GREEDY] Routing N/total...`) demoted to DEBUG

## What Remains

- `@profile_time` not yet applied to core algorithm functions in `unified_pathfinder.py`
- No conditional `if condition:` guards before expensive log format calls in hot paths
- Per-net ROI messages (512×/iteration) still at DEBUG — add guards if file size still excessive after re-test
