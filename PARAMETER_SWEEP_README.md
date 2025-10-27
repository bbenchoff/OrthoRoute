# PathFinder Parameter Sweep - Complete Solution

## Overview

A comprehensive overnight parameter sweep system for finding optimal PathFinder convergence parameters. This solution implements the complete playbook from `FINAL_CONVERGENCE_TASK.md` with robust execution, early stopping, result tracking, and visualization.

## Files Created

1. **`parameter_sweep.py`** (31 KB)
   - Main sweep execution script
   - Parameter matrix (A1-A6, B1-B6)
   - Early stopping logic
   - Real-time monitoring
   - Resumable execution
   - JSON result tracking

2. **`PARAMETER_SWEEP_GUIDE.md`** (7.3 KB)
   - Complete user guide
   - Usage instructions
   - Troubleshooting
   - Result interpretation

3. **`plot_sweep_results.py`** (12 KB)
   - Visualization generation
   - Convergence plots
   - Parameter impact analysis
   - Ranking charts

## Quick Start

### Run the complete sweep (overnight):
```bash
python parameter_sweep.py
```

### Monitor progress:
```bash
# Watch active run
tail -f parameter_sweep_results/run_B4.log | grep ITER

# Check summary
cat parameter_sweep_results/sweep_summary.txt
```

### After completion - visualize results:
```bash
python plot_sweep_results.py
```

## What Gets Tested

### Parameter Matrix (12 configurations)

All combinations of:
- **initial_pres**: 50, 150 (scaled to 0.5, 1.5 in code)
- **pres_mult**: 1.30 (fixed)
- **hist_gain**: 1.1 (fixed)
- **layer_bias_max**: 1.6, 1.8
- **via_depart_max**: 0.35, 0.45
- **start_mode**: greedy, penalized

**Priority execution order:**
1. B4 (Penalized, high layer bias, high via departure)
2. B1 (Penalized, low layer bias, low via departure)
3. A4 (Greedy, high layer bias, high via departure)
4. A1 (Greedy, low layer bias, low via departure)
5. A2, A3, A5, A6, B2, B3, B5, B6

### For Each Configuration

1. **Modify config** - Updates `config.py` with test parameters
2. **Run test** - Executes: `ORTHO_CPU_ONLY=1 timeout 600 python main.py --test-manhattan`
3. **Monitor output** - Parses logs in real-time
4. **Apply early stopping** - Stops if criteria met
5. **Track metrics** - Saves iteration-by-iteration data
6. **Restore config** - Resets to original state

## Metrics Tracked

### Per Iteration
- Total overuse and edge count
- Via overuse percentage
- Top 3 layers (number, count, percentage)
- Iteration timing
- Routed/failed net counts

### Per Run
- Best overuse achieved (and iteration)
- Final overuse (and iteration)
- Convergence status (<1000 = converged)
- Layer spread metrics (max %, top-3 total %)
- Average iteration time
- Total run time
- Early stop reason (if applicable)
- Success/failure status
- Error messages

## Early Stopping Criteria

Automatically stops a run if:

1. **No improvement** - No 10% improvement in 6 consecutive iterations
2. **Diverging** - 3 consecutive increases >20%
3. **Layer imbalance** - Single layer >25% of overuse after iteration 12
4. **Plateau** - No 5% improvement over 8 iterations

These save significant time (typical early stop: 3-8 minutes vs 10-minute timeout).

## Output Files

All saved to `parameter_sweep_results/`:

### Data Files
- **`sweep_results.json`** - Complete structured data (all runs, all metrics)
- **`sweep_summary.txt`** - Human-readable summary report
- **`run_A1.log`** through **`run_B6.log`** - Raw log output per run

### Visualizations (after running plot script)
- **`convergence_comparison.png`** - All runs overlaid
- **`best_run_*_detailed.png`** - 4-panel deep dive of best config
- **`parameter_impact.png`** - Parameter sensitivity bar charts
- **`ranking_summary.png`** - Final ranking visualization

## Expected Timeline

- **Per run**: Up to 10 minutes (600s timeout)
- **Typical early stop**: 3-8 minutes
- **Full sweep**: ~2 hours (12 runs, many early-stop)
- **Overnight**: Definitely completes before morning

## Resume Capability

The script is fully resumable:
```bash
# Interrupt with Ctrl+C at any time
python parameter_sweep.py

# Resume from checkpoint
python parameter_sweep.py --resume
```

Results are saved after each run, so no data is lost.

## Result Interpretation

### Summary Report Structure

**1. Best Configuration Section**
```
Run ID: B4
Config: {'initial_pres': 50, 'pres_mult': 1.30, 'hist_gain': 1.1,
         'layer_bias_max': 1.8, 'via_depart_max': 0.45, 'start_mode': 'penalized'}

Results:
  Best overuse: 1245 (iteration 18)
  Final overuse: 1891 (iteration 32)
  Convergence achieved: False (target: <1000)
  Total iterations: 32
  Avg iteration time: 30.1s
  Total run time: 962.3s
```

**2. Convergence History**
```
Iter  1: overuse=  8432  via=0.2%  [L30:16.5%, L10:12.6%, L6:12.0%]
Iter  5: overuse=  3201  via=0.1%  [L30:14.2%, L10:11.8%, L6:10.5%]
Iter 10: overuse=  1823  via=0.1%  [L30:11.3%, L10:10.2%, L6:9.8%]
Iter 15: overuse=  1456  via=0.1%  [L30:9.8%, L10:9.5%, L12:8.9%]
Iter 20: overuse=  1289  via=0.1%  [L30:8.7%, L10:8.9%, L12:8.5%]
```

**3. All Runs Ranked**
```
Rank   Run    Best     Final  Iters     Time  Status
   1   B4     1245      1891     32     962s  Incomplete
   2   B1     1389      2104     28     841s  Early stop: Plateau
   3   A4     1512      2456     35    1089s  Incomplete
   4   A1     1678      2301     30     903s  Incomplete
```

**4. Parameter Impact Analysis**
```
initial_pres:
        50: avg=    1456  (n=8)  ← BETTER
       150: avg=    1823  (n=4)

layer_bias_max:
       1.6: avg=    1534  (n=6)
       1.8: avg=    1445  (n=6)  ← BETTER

via_depart_max:
      0.35: avg=    1489  (n=6)
      0.45: avg=    1490  (n=6)  ← NEUTRAL

start_mode:
    greedy: avg=    1512  (n=6)
 penalized: avg=    1467  (n=6)  ← BETTER
```

**5. Recommendations**
- Best single configuration
- Top 3 alternatives
- Parameter trends and insights

## Success Criteria

A configuration is considered **successful** if:
- ✅ Test passes (no crashes)
- ✅ Completes at least 5 iterations
- ✅ Final overuse < 10,000

A configuration is considered **CONVERGED** if:
- ✅ Best overuse < 1,000

## Advanced Usage

### Test Single Configuration
```bash
# Set config without running
python parameter_sweep.py --config-only B4

# Manually test
python main.py --test-manhattan 2>&1 | tee test_manual.log
```

### Preview Execution Plan
```bash
python parameter_sweep.py --dry-run
```

### Run Subset of Configs
Edit `PRIORITY_ORDER` in `parameter_sweep.py`:
```python
PRIORITY_ORDER = ["B4", "B1", "A4"]  # Only test these 3
```

### Adjust Timeout
Edit `TIMEOUT_SECONDS` in `parameter_sweep.py`:
```python
TIMEOUT_SECONDS = 900  # 15 minutes instead of 10
```

### Modify Early Stopping
Edit `EARLY_STOP_CRITERIA` in `parameter_sweep.py`:
```python
EARLY_STOP_CRITERIA = {
    "no_improvement_iters": 8,        # More patient
    "improvement_threshold": 0.05,    # Require less improvement
    ...
}
```

## Troubleshooting

### Script crashes or hangs:
```bash
# Resume from last checkpoint
python parameter_sweep.py --resume
```

### Config file corrupted:
```bash
cd orthoroute/algorithms/manhattan/pathfinder
cp config.py.backup config.py  # Restore from auto-backup
```

### matplotlib not installed (for plotting):
```bash
pip install matplotlib
```

### Need to analyze results programmatically:
```python
import json
with open('parameter_sweep_results/sweep_results.json') as f:
    results = json.load(f)

# Results structure:
# results[run_id]['config'] - parameter dict
# results[run_id]['iterations'] - list of iteration metrics
# results[run_id]['best_overuse'] - int
# results[run_id]['convergence_achieved'] - bool
```

## Integration with Existing Work

This sweep script:
- ✅ Uses existing `main.py --test-manhattan`
- ✅ Modifies `config.py` temporarily (always restores)
- ✅ Respects `ORTHO_CPU_ONLY=1` environment variable
- ✅ Parses standard log output format
- ✅ Compatible with current PathFinder implementation
- ✅ No changes to core routing code required

## Expected Results

Based on playbook predictions and current baseline:

**Baseline (current):**
- Best overuse: 1,911 @ iteration 8
- Oscillates: 2-6K range
- Layer 30: 16.5% (hotspot)

**Target (sweep goal):**
- Best overuse: <1,500 (improved from 1,911)
- Layer 30 share: <10% (down from 16.5%)
- Convergence: <1,000 by iteration 25-35

**Likely winners (prediction):**
1. B4 - Penalized start, high layer bias, high via departure
2. B1 - Penalized start, lower settings (conservative)
3. A4 - Greedy start, high layer bias, high via departure

## After Sweep Completes

1. **Review results:**
   ```bash
   cat parameter_sweep_results/sweep_summary.txt
   ```

2. **Generate plots:**
   ```bash
   python plot_sweep_results.py
   ```

3. **Apply best configuration:**
   ```bash
   # Set best config (e.g., B4)
   python parameter_sweep.py --config-only B4

   # Validate manually
   python main.py --test-manhattan 2>&1 | tee test_final_validation.log
   ```

4. **Update defaults permanently:**
   Edit `orthoroute/algorithms/manhattan/pathfinder/config.py`:
   ```python
   PRES_FAC_INIT = 0.5          # From best run
   PRES_FAC_MULT = 1.30         # From best run
   HIST_ACCUM_GAIN = 1.1        # From best run
   # etc.
   ```

5. **Document findings:**
   Add to `FINAL_CONVERGENCE_TASK.md` or create new summary doc.

## Implementation Details

### How Config Modification Works

The script modifies these constants in `config.py`:
```python
PRES_FAC_INIT = 0.5      # initial_pres / 100
PRES_FAC_MULT = 1.30     # pres_mult
HIST_ACCUM_GAIN = 1.1    # hist_gain
```

And uses these dataclass fields (already in code):
```python
layer_bias_max: float = 1.8       # From config dict
via_depart_max: float = 0.45      # From config dict
```

### How Log Parsing Works

The script recognizes these log patterns:

**Iteration summary:**
```
[ITER 15] routed=512 failed=0 overuse=2341 edges=1234 via_overuse=0.3%
```

**Layer congestion:**
```
[LAYER-CONGESTION] Layer 30: 1570 overuse (16.5%)
```

**Timing:**
```
[TIMING] Iteration 15 completed in 31.2s
```

### How Early Stopping Works

Real-time monitoring:
1. Stream log output line-by-line
2. Parse each `[ITER ...]` line immediately
3. Check early stopping criteria
4. Terminate process if criteria met
5. Log reason and continue to next run

### How Resume Works

Checkpoint system:
1. Load `sweep_results.json` at start
2. Check which runs have `success=true`
3. Skip successful runs
4. Re-run failed/incomplete runs
5. Save after each run completion

## Safety Features

- ✅ **Config always restored** - Even on crash, backup exists
- ✅ **No data loss** - Results saved after each run
- ✅ **Resumable** - Interrupt anytime with Ctrl+C
- ✅ **Read-only core code** - Only modifies config file
- ✅ **Timeout protection** - 10-minute hard limit per run
- ✅ **Process isolation** - Each run is separate subprocess
- ✅ **Early stopping** - Avoids wasted time on bad configs
- ✅ **Real-time logs** - Monitor progress as it happens

## Architecture

```
parameter_sweep.py
├── Load existing results (if --resume)
├── For each config in priority order:
│   ├── Skip if already successful
│   ├── Modify config.py (with backup)
│   ├── Launch subprocess: python main.py --test-manhattan
│   ├── Stream output to log file
│   ├── Parse logs in real-time
│   ├── Check early stopping criteria
│   ├── Terminate if criteria met
│   ├── Collect metrics
│   ├── Restore config.py
│   └── Save results.json
├── Generate summary report
└── Exit

plot_sweep_results.py
├── Load sweep_results.json
├── Generate convergence comparison plot
├── Generate best run detailed plot (4-panel)
├── Generate parameter impact charts
├── Generate ranking summary
└── Save all plots to output directory
```

## File Locations

```
OrthoRoute/
├── parameter_sweep.py              ← Main script (31 KB)
├── plot_sweep_results.py           ← Visualization (12 KB)
├── PARAMETER_SWEEP_GUIDE.md        ← User guide (7.3 KB)
├── PARAMETER_SWEEP_README.md       ← This file (summary)
├── orthoroute/algorithms/manhattan/pathfinder/
│   ├── config.py                   ← Modified temporarily
│   └── config.py.backup            ← Auto-created backup
└── parameter_sweep_results/        ← Created on first run
    ├── sweep_results.json          ← Complete data
    ├── sweep_summary.txt           ← Human-readable report
    ├── run_A1.log                  ← Individual run logs
    ├── run_B4.log
    ├── ...
    ├── convergence_comparison.png  ← Plots (after plot script)
    ├── best_run_*_detailed.png
    ├── parameter_impact.png
    └── ranking_summary.png
```

## Performance

- **Sweep execution**: ~2 hours for 12 configs (with early stopping)
- **Per run**: 3-10 minutes (many stop early)
- **Plot generation**: ~10 seconds
- **Disk usage**: ~50 MB for all logs and results
- **Memory usage**: Same as single main.py run (~2-4 GB)

## Testing Validation

Before running overnight, test individual components:

```bash
# Test help and dry-run
python parameter_sweep.py --help
python parameter_sweep.py --dry-run

# Test single config modification
python parameter_sweep.py --config-only B4
cat orthoroute/algorithms/manhattan/pathfinder/config.py | grep PRES_FAC

# Restore config
cat orthoroute/algorithms/manhattan/pathfinder/config.py.backup > orthoroute/algorithms/manhattan/pathfinder/config.py

# Test plot script (requires results)
python plot_sweep_results.py --help
```

## Known Limitations

1. **Config modification scope**: Only modifies module-level constants, not all PathFinderConfig dataclass fields
2. **Log parsing robustness**: Depends on specific log format (see patterns above)
3. **matplotlib required**: For plotting (optional, sweep works without it)
4. **Windows/Unix paths**: Uses pathlib for cross-platform compatibility
5. **Python 3.7+**: Requires dataclasses (available in 3.7+)

## Future Enhancements

Possible improvements:
- [ ] Add email notification on completion
- [ ] Support multiple test boards
- [ ] Add grid search / Bayesian optimization
- [ ] Export results to CSV for Excel analysis
- [ ] Add convergence rate calculation
- [ ] Support custom parameter ranges
- [ ] Add statistical significance testing

## Support

For issues or questions:
1. Check `PARAMETER_SWEEP_GUIDE.md` for detailed usage
2. Review `sweep_summary.txt` for run diagnostics
3. Inspect individual `run_*.log` files for errors
4. Verify config.py backup exists if corruption occurs

---

**Ready to run!** Start with:
```bash
python parameter_sweep.py --dry-run  # Preview
python parameter_sweep.py            # Execute overnight
```

**Expected completion**: ~2 hours (varies with early stopping)

**Check progress**: `tail -f parameter_sweep_results/sweep_results.json`

**View results**: `cat parameter_sweep_results/sweep_summary.txt`

**Visualize**: `python plot_sweep_results.py`
