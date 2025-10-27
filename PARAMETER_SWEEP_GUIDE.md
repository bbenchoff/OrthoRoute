# Parameter Sweep Guide

## Overview

The `parameter_sweep.py` script automates overnight testing of PathFinder parameter configurations to find optimal convergence settings.

## Quick Start

### Run the full sweep (overnight):
```bash
python parameter_sweep.py
```

### Resume interrupted sweep:
```bash
python parameter_sweep.py --resume
```

### See execution plan without running:
```bash
python parameter_sweep.py --dry-run
```

### Test a single configuration:
```bash
python parameter_sweep.py --config-only B4
python main.py --test-manhattan  # Run with B4 config
```

## What It Does

### 1. Parameter Matrix (12 configurations)

**Greedy-start runs (A1-A6):**
- A1: initial_pres=50, layer_bias_max=1.6, via_depart_max=0.35
- A2: initial_pres=50, layer_bias_max=1.6, via_depart_max=0.45
- A3: initial_pres=50, layer_bias_max=1.8, via_depart_max=0.35
- A4: initial_pres=50, layer_bias_max=1.8, via_depart_max=0.45
- A5: initial_pres=150, layer_bias_max=1.6, via_depart_max=0.35
- A6: initial_pres=150, layer_bias_max=1.6, via_depart_max=0.45

**Penalized-start runs (B1-B6):**
- Same parameters as A1-A6 but with penalized start mode

**Priority Order:** B4, B1, A4, A1, A2, A3, A5, A6, B2, B3, B5, B6

### 2. For Each Configuration

The script:
- Modifies `orthoroute/algorithms/manhattan/pathfinder/config.py`
- Runs: `ORTHO_CPU_ONLY=1 timeout 600 python main.py --test-manhattan`
- Monitors output in real-time
- Applies early stopping if criteria met
- Restores original config file

### 3. Early Stopping Criteria

Automatically stops a run if:
- **No improvement:** No 10% improvement in 6 iterations
- **Diverging:** 3 consecutive increases >20%
- **Layer imbalance:** Single layer >25% after iteration 12
- **Plateau:** No 5% improvement over 8 iterations

### 4. Metrics Tracked

**Per iteration:**
- Total overuse and edges with overuse
- Via overuse percentage
- Top 3 layers by overuse (layer number, count, percentage)
- Iteration timing
- Routed/failed net counts

**Per run:**
- Best overuse and iteration achieved
- Final overuse and iteration
- Convergence status (<1000 overuse = converged)
- Layer spread metrics (max layer %, top-3 total %)
- Average iteration time
- Early stop reason (if applicable)

### 5. Output Files

All results stored in `parameter_sweep_results/`:

- **sweep_results.json** - Complete structured data (all metrics)
- **sweep_summary.txt** - Human-readable summary report
- **run_A1.log, run_B4.log, etc.** - Raw log output per run

## Expected Timeline

- **Per run:** Up to 10 minutes (600s timeout)
- **Early stops:** Typically 3-8 minutes
- **Full sweep:** ~2 hours (12 runs, many will early-stop)
- **Overnight safe:** Yes, script is robust and resumable

## Interpreting Results

### Summary Report Sections

**1. Best Configuration**
```
Run ID: B4
Config: {'initial_pres': 50, 'pres_mult': 1.30, ...}
Best overuse: 1245 (iteration 18)
Final overuse: 1891 (iteration 32)
Convergence achieved: False
```

**2. Convergence History**
```
Iter  1: overuse=  8432  via=0.2%  [L30:16.5%, L10:12.6%, L6:12.0%]
Iter  5: overuse=  3201  via=0.1%  [L30:14.2%, L10:11.8%, L6:10.5%]
Iter 10: overuse=  1823  via=0.1%  [L30:11.3%, L10:10.2%, L6:9.8%]
```

**3. Ranking Table**
```
Rank   Run    Best     Final  Iters     Time  Status
   1   B4     1245      1891     32     962s  Incomplete
   2   B1     1389      2104     28     841s  Early stop: Plateau
   3   A4     1512      2456     35    1089s  Incomplete
```

**4. Parameter Impact Analysis**
```
initial_pres:
        50: avg=    1456  (n=8)
       150: avg=    1823  (n=4)

layer_bias_max:
       1.6: avg=    1534  (n=6)
       1.8: avg=    1445  (n=6)
```

**5. Recommendations**
- Best single configuration
- Top 3 alternatives
- Parameter trends

## Monitoring Progress

### While running:
```bash
# Watch overall progress
tail -f parameter_sweep_results/sweep_results.json

# Monitor specific run
tail -f parameter_sweep_results/run_B4.log | grep ITER

# Check convergence in real-time
watch -n 10 "grep 'ITER.*overuse' parameter_sweep_results/run_B4.log | tail -5"
```

### Check completion:
```bash
cat parameter_sweep_results/sweep_summary.txt
```

## Troubleshooting

### Script hangs or crashes:
```bash
# Resume from checkpoint
python parameter_sweep.py --resume
```

### Config file corruption:
The script automatically creates `.backup` files and restores on exit.
If needed, manually restore:
```bash
cd orthoroute/algorithms/manhattan/pathfinder
cp config.py.backup config.py
```

### Test single config before sweep:
```bash
python parameter_sweep.py --config-only A1
python main.py --test-manhattan 2>&1 | tee test_manual.log
```

### Change timeout (default 600s):
Edit `parameter_sweep.py` line:
```python
TIMEOUT_SECONDS = 600  # Increase for longer runs
```

### Adjust early stopping criteria:
Edit `EARLY_STOP_CRITERIA` dictionary in `parameter_sweep.py`:
```python
EARLY_STOP_CRITERIA = {
    "no_improvement_iters": 6,        # Increase to be more patient
    "improvement_threshold": 0.10,    # Lower to require less improvement
    ...
}
```

## Advanced Usage

### Run subset of configurations:
Edit `PRIORITY_ORDER` in script to include only desired runs:
```python
PRIORITY_ORDER = ["B4", "B1", "A4"]  # Only run these 3
```

### Add custom configurations:
Add to `PARAMETER_MATRIX`:
```python
"C1": {"initial_pres": 75, "pres_mult": 1.35, "hist_gain": 1.2,
       "layer_bias_max": 1.7, "via_depart_max": 0.40, "start_mode": "greedy"},
```

And add `"C1"` to `PRIORITY_ORDER`.

### Export results for analysis:
```python
import json
with open('parameter_sweep_results/sweep_results.json') as f:
    results = json.load(f)

# Analyze with pandas, matplotlib, etc.
```

## Success Criteria

A configuration is considered successful if:
- Test passes (no crashes)
- Completes at least 5 iterations
- Final overuse < 10,000
- Best overuse < 1,000 = **CONVERGED** ✅

## Expected Best Results

Based on playbook predictions:

**Likely winners:**
1. B4 (Penalized, initial_pres=50, layer_bias_max=1.8, via_depart_max=0.45)
2. B1 (Penalized, initial_pres=50, layer_bias_max=1.6, via_depart_max=0.35)
3. A4 (Greedy, initial_pres=50, layer_bias_max=1.8, via_depart_max=0.45)

**Target metrics:**
- Best overuse: <1,500 (improved from 1,911 baseline)
- Layer 30 share: <10% (down from 16.5%)
- Convergence: <1,000 overuse by iteration 25-35

## Notes

- Script is **resumable** - safe to interrupt with Ctrl+C
- Config file is **always restored** (even on crash)
- Results are **saved after each run** (no data loss)
- Logs are **streamed in real-time** (monitor progress)
- Early stopping **saves time** (no wasted runs)

## After Sweep Completes

1. Review `parameter_sweep_results/sweep_summary.txt`
2. Identify best configuration
3. Apply best config permanently:
   ```bash
   python parameter_sweep.py --config-only B4  # or whatever won
   ```
4. Test best config manually:
   ```bash
   python main.py --test-manhattan 2>&1 | tee test_best_config.log
   ```
5. Update `config.py` defaults to best values

---

**Ready to run!** Start with:
```bash
python parameter_sweep.py --dry-run  # Preview
python parameter_sweep.py            # Execute
```
