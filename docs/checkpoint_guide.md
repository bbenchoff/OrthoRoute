# OrthoRoute Checkpoint System

## Overview

The checkpoint system allows you to save and restore routing state at any iteration. This is useful for:

1. **Crash Recovery**: Resume routing after a crash or interruption
2. **Parameter Tuning**: Load a checkpoint and continue with different parameters
3. **Experimentation**: Try different strategies from the same starting point

## Features

### Auto-Save (Default: Enabled)
- Automatically saves a checkpoint after **every iteration**
- Keeps the **last 10 checkpoints** (older ones are automatically deleted)
- Checkpoints saved to: `checkpoints/checkpoint_iterXXX_TIMESTAMP.pkl`

### Manual Save/Load
- **Ctrl+S**: Save checkpoint manually
- **Ctrl+L**: Load checkpoint from file browser
- **Ctrl+R**: Resume from latest checkpoint

### Checkpoint Contents

Each checkpoint contains:
- Current iteration number
- All routed net paths
- Congestion state (present, history, present_ema)
- Via usage state
- Routing parameters (pres_fac, config)
- Metadata (overuse, routed nets, failed nets)

## Usage

### Via GUI

1. **File > Checkpoints > Auto-save Checkpoints** (default: ON)
   - Automatically saves after each iteration

2. **File > Checkpoints > Save Checkpoint Now** (Ctrl+S)
   - Manually save current state

3. **File > Checkpoints > Load Checkpoint...** (Ctrl+L)
   - Browse and load a specific checkpoint
   - View iteration, overuse, and routing stats

4. **File > Checkpoints > Resume from Latest** (Ctrl+R)
   - Quickly load the most recent checkpoint

### Via Python API

```python
from orthoroute.algorithms.manhattan.unified_pathfinder import PathFinderRouter

# Create router
router = PathFinderRouter()

# Enable/disable auto-checkpoint
router.auto_checkpoint = True  # Default

# Manual save
router.checkpoint_manager.save_checkpoint(
    router,
    iteration=5,
    pres_fac=3.3,
    metadata={'note': 'Good state before parameter change'}
)

# Load checkpoint
checkpoint = router.load_checkpoint('checkpoints/checkpoint_iter005_20251104_120000.pkl')

# Resume from latest
router.resume_from_checkpoint()  # Uses latest checkpoint

# List all checkpoints
checkpoints = router.checkpoint_manager.list_checkpoints()
```

## Use Cases

### 1. Crash Recovery
If routing crashes at iteration 15:
```
1. Restart OrthoRoute
2. File > Checkpoints > Resume from Latest (Ctrl+R)
3. Continue routing from iteration 15
```

### 2. Parameter Tuning
You want to try more aggressive pressure after iteration 10:
```
1. Let routing run to iteration 10
2. File > Checkpoints > Load Checkpoint... (select iter010)
3. Modify parameters:
   - Edit parameter_derivation.py: pres_fac_mult = 2.0
4. Continue routing with new parameters
```

### 3. Comparison Testing
Test two different strategies from the same starting point:
```
1. Route to iteration 5
2. Save: File > Checkpoints > Save Checkpoint Now
3. Try strategy A (e.g., high via cost)
4. File > Checkpoints > Load Checkpoint... (back to iter 5)
5. Try strategy B (e.g., low via cost)
6. Compare results
```

## Checkpoint File Size

Typical checkpoint sizes:
- **Small board (512 nets)**: ~5-10 MB
- **Medium board (2K nets)**: ~20-50 MB
- **Large board (8K nets)**: ~100-200 MB

With auto-save keeping 10 checkpoints, expect **1-2 GB** disk usage for large boards.

## Technical Details

### What's Saved
- `net_paths`: Dict[str, List[int]] - All routed paths
- `accounting.present`: Current edge usage
- `accounting.present_ema`: Smoothed present costs
- `accounting.history`: Historical congestion
- `accounting.capacity`: Edge capacities
- `via_col_use`, `via_seg_use`: Via usage state
- `iteration`, `pres_fac`: Routing state
- Metadata: overuse, routed_nets, failed_nets, etc.

### What's NOT Saved
- Board geometry (lattice, graph) - reconstructed from board
- GPU kernel state - recompiled on load
- Progress callbacks - set up on resume

### GPU Arrays
GPU arrays (CuPy) are automatically converted to CPU (NumPy) when saving, and restored to GPU when loading if GPU is enabled.

## Troubleshooting

### "No checkpoints found"
- Check `checkpoints/` directory exists
- Routing must run at least 1 iteration to create checkpoint

### "Failed to load checkpoint"
- Checkpoint file may be corrupted
- Try loading an earlier checkpoint

### Large checkpoint files
- Adjust keep_last_n in `cleanup_old_checkpoints(keep_last_n=5)` to keep fewer checkpoints
- Manually delete old checkpoints from `checkpoints/` directory

## Disabling Auto-Save

If you don't want auto-save (e.g., for very large boards):
```python
router.auto_checkpoint = False
```

Or via GUI:
```
File > Checkpoints > Auto-save Checkpoints (uncheck)
```
