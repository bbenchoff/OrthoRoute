# Quick Start: GPU Hardcoded Configuration

## TL;DR - What Changed

**ALL GPU environment variables have been removed and replaced with hardcoded values in `GPUConfig` class.**

No more setting environment variables before running the router. Everything is now controlled via code.

---

## Quick Test (30 seconds)

```bash
# Test 1: Verify imports work
python -c "from orthoroute.algorithms.manhattan.unified_pathfinder import GPUConfig; print('GPU_MODE:', GPUConfig.GPU_MODE)"

# Expected output:
# GPU_MODE: True
```

If this works, you're good to go!

---

## What's Enabled by Default

```python
GPU_MODE = True                    # GPU batching ON
GPU_PERSISTENT_ROUTER = True       # Agent B1 persistent kernel ON
GPU_DEVICE_COMPACTION = True       # Device-side compaction ON
GPU_DEVICE_ROI = True              # ROI bounding boxes ON
GPU_DEVICE_ACCOUNTING = True       # GPU cost updates ON
USE_ASTAR = True                   # A* heuristic ON
```

**Translation:** Full GPU optimization stack is enabled out-of-the-box.

---

## How to Change Settings

### Option 1: Edit GPUConfig class (Permanent)

**File:** `orthoroute/algorithms/manhattan/unified_pathfinder.py`
**Line:** 527

```python
class GPUConfig:
    GPU_MODE = False  # Disable GPU (use CPU only)
    # ... other settings
```

### Option 2: Runtime Override (Temporary)

```python
from orthoroute.algorithms.manhattan.unified_pathfinder import GPUConfig

# Disable GPU for this run only
GPUConfig.GPU_MODE = False

# Then create router
from orthoroute.algorithms.manhattan.unified_pathfinder import UnifiedPathFinder
router = UnifiedPathFinder()
```

---

## Common Use Cases

### Disable GPU Entirely
```python
GPUConfig.GPU_MODE = False
```

### Disable Persistent Router (use iterative kernel)
```python
GPUConfig.GPU_PERSISTENT_ROUTER = False
```

### Enable Delta-Stepping (if memory bug is fixed)
```python
GPUConfig.USE_DELTA_STEPPING = True
```

### Adjust Capacity for Testing
```python
GPUConfig.CAPACITY_MULT = 2.0  # Double all edge capacities
```

---

## Where to Find Things

| What | Where |
|------|-------|
| GPUConfig class | `orthoroute/algorithms/manhattan/unified_pathfinder.py:527` |
| Full documentation | `GPU_HARDCODED_CONFIG_REPORT.md` |
| All config options | See GPUConfig class definition |

---

## Verification Commands

```bash
# Check no GPU env vars remain in algorithm files
grep -r "os.environ.get" orthoroute/algorithms/manhattan/*.py orthoroute/algorithms/manhattan/pathfinder/*.py | grep -v ".bak"
# Expected: (empty)

# Test full import chain
python -c "from orthoroute.algorithms.manhattan.unified_pathfinder import GPUConfig, UnifiedPathFinder, PathFinderConfig; from orthoroute.algorithms.manhattan.pathfinder.cuda_dijkstra import CUDADijkstra; print('SUCCESS')"
# Expected: SUCCESS
```

---

## Files Modified

1. `orthoroute/algorithms/manhattan/unified_pathfinder.py` - Added GPUConfig class, removed 3 env var checks
2. `orthoroute/algorithms/manhattan/pathfinder/cuda_dijkstra.py` - Added GPUConfig import, removed 3 env var checks
3. `orthoroute/presentation/gui/main_window.py` - Added GPUConfig imports, removed 3 env var checks

**Total:** 9 environment variable calls removed, 0 compilation errors.

---

## Environment Variables That Still Exist (Non-GPU)

These are intentionally kept:
- `KICAD_API_SOCKET` - Required for KiCad IPC
- `KICAD_API_TOKEN` - Required for KiCad auth
- `ORTHO_NO_SCREENSHOTS` - GUI debug feature
- `ORTHO_SCREENSHOT_FREQ` - GUI debug feature
- `ORTHO_SCREENSHOT_SCALE` - GUI debug feature

**None of these affect GPU routing.**

---

## Status: READY FOR TESTING

All GPU configuration is now hardcoded. Plugin is KiCad-compatible.

**Next step:** Test routing in your KiCad plugin environment.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'orthoroute'"
Make sure you're in the project root directory or have the package installed.

### "ImportError: cannot import name 'GPUConfig'"
Old cached bytecode. Delete `__pycache__` directories:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

### "AttributeError: 'GPUConfig' object has no attribute 'XYZ'"
Check the GPUConfig class definition - you may be accessing a removed/renamed attribute.

---

## Quick Reference: GPUConfig Attributes

```python
# Core features
GPU_MODE                    # Master GPU enable
GPU_PERSISTENT_ROUTER       # Persistent kernel
GPU_DEVICE_COMPACTION       # Device compaction
GPU_DEVICE_ROI              # ROI bounding
GPU_DEVICE_ACCOUNTING       # GPU cost updates
GPU_DEVICE_BACKTRACE        # Device backtrace

# Algorithms
USE_DELTA_STEPPING          # Delta-stepping algorithm
USE_BIDIR_ASTAR             # Bidirectional A*
USE_ASTAR                   # A* heuristic

# Debug/testing
DEBUG_INVARIANTS            # CSR validation
CAPACITY_MULT               # Capacity multiplier
DELTA_STEPPING_DELTA        # Delta parameter
ENABLE_AUTO_DELTA           # Auto algorithm switch
```

---

**For full details, see `GPU_HARDCODED_CONFIG_REPORT.md` (15KB comprehensive report).**
