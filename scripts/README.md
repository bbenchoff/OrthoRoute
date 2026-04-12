# OrthoRoute Scripts

Automation scripts for optimization workflows and log analysis.

---

## **Scripts Overview**

| Script | Purpose | Usage |
|--------|---------|-------|
| [analyze_log.py](#analyze_logpy) | Parse routing logs, extract metrics, compare with golden thresholds | `python scripts/analyze_log.py [options]` |
| [optimize_and_validate.ps1](#optimize_and_validateps1) | Automated optimization cycle: deploy → test → validate | `.\scripts\optimize_and_validate.ps1 [options]` |

---

## **analyze_log.py**

Standalone log parser for OrthoRoute routing logs. Extracts routing metrics, convergence status, and performance data.

### **Basic Usage**

```powershell
# Analyze latest run (logs/latest.log)
python scripts/analyze_log.py

# Analyze specific log file
python scripts/analyze_log.py --log-file logs/run_20260410_184636.log

# Output as JSON
python scripts/analyze_log.py --json

# Compare against golden metrics
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json

# Export comparison as JSON for automation
python scripts/analyze_log.py --compare tests/regression/smoke_metrics.json --json > results.json
```

### **Output Format**

**Human-readable** (default):
```
================================================================================
OrthoRoute Log Analysis
================================================================================

ROUTING SUMMARY
--------------------------------------------------------------------------------
  Nets routed:       512/512
  Converged:         True
  Iterations:        73
  Total time:        1106.6s (18.4 min)
  Avg iteration:     15.2s
  Overuse edges:     0
  Barrel conflicts:  367
  Tracks written:    4274
  Vias written:      2738
  GPU mode:          YES

LATTICE DIMENSIONS
--------------------------------------------------------------------------------
  Grid:              106×234×18
  Total nodes:       446,472

PROFILING DATA (Top 10 by total time)
--------------------------------------------------------------------------------
  _build_owner_bitmap_for_fullgraph              67.2s  (  73 calls,   920.0ms avg)
  commit_path                                     8.5s  ( 512 calls,    16.6ms avg)
  ...

GOLDEN COMPARISON
--------------------------------------------------------------------------------
  Overall status: PASS
  Mode:           GPU

  ✓ nets_routed         actual=     512 expected=     512 => PASS
  ✓ total_nets          actual=     512 expected=     512 => PASS
  ✓ converged           actual=    True expected=    True => PASS
  ✓ iterations          actual=      73 threshold=      88 => PASS
  ✓ total_time_s        actual=  1106.6 threshold=  1328.0 => PASS
  ✓ barrel_conflicts    actual=     367 threshold=     450 => PASS

================================================================================
```

**JSON** (`--json`):
```json
{
  "routing_summary": {
    "success": true,
    "converged": true,
    "nets_routed": 512,
    "total_nets": 512,
    "iterations": 73,
    "total_time_s": 1106.6,
    "barrel_conflicts": 367,
    "tracks_written": 4274,
    "vias_written": 2738,
    ...
  },
  "lattice": {
    "cols": 106,
    "rows": 234,
    "layers": 18,
    "nodes": 446472
  },
  "gpu_mode": true,
  "profiling": {
    "_build_owner_bitmap_for_fullgraph": {
      "total_ms": 67200.0,
      "count": 73,
      "avg_ms": 920.0,
      ...
    }
  },
  "comparison": {
    "overall_status": "PASS",
    "mode": "gpu",
    "checks": [...]
  }
}
```

### **Golden Comparison**

When using `--compare`, the script validates metrics against thresholds:

| Status | Meaning | Exit Code |
|--------|---------|-----------|
| **PASS** | All checks passed | 0 |
| **WARN** | Performance regression (soft warnings) | 0 |
| **FAIL** | Hard failure (e.g., nets not routed, not converged) | 1 |

**Required checks** (hard failures):
- `nets_routed` must equal `total_nets`
- `converged` must be `True`

**Performance checks** (soft warnings):
- `iterations` ≤ threshold
- `total_time_s` ≤ threshold
- `barrel_conflicts` ≤ threshold

### **Integration Examples**

**In automation/CI pipelines:**
```powershell
# Run routing and validate
pytest tests/regression/test_smoke.py -v
$metrics = python scripts/analyze_log.py --compare tests/regression/smoke_metrics.json --json | ConvertFrom-Json

if ($metrics.comparison.overall_status -eq 'FAIL') {
    Write-Host "Routing regression detected!"
    exit 1
}
```

**Quick performance check:**
```powershell
# After making a code change
python scripts/analyze_log.py --compare tests/regression/smoke_metrics.json

# Look for WARN or FAIL in output
```

---

## **optimize_and_validate.ps1**

Automated optimization workflow that streamlines the edit → deploy → test → validate cycle.

### **Basic Usage**

```powershell
# Quick smoke test (100 nets, <30s)
.\scripts\optimize_and_validate.ps1

# Full validation with profiling and golden comparison
.\scripts\optimize_and_validate.ps1 -ProfileMode -Compare tests/regression/smoke_metrics.json

# Full backplane test (512 nets, 11-18 min) without re-deploying
.\scripts\optimize_and_validate.ps1 -TestBoard backplane -SkipDeploy

# Export results as JSON
.\scripts\optimize_and_validate.ps1 -Json > results.json

# Show full log after test
.\scripts\optimize_and_validate.ps1 -ShowLog
```

### **Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-TestBoard` | Which test to run: `smoke` (100 nets) or `backplane` (512 nets) | `smoke` |
| `-SkipDeploy` | Skip `copy_to_kicad.ps1` sync step | (off) |
| `-ProfileMode` | Enable `ORTHO_DEBUG=1` for detailed profiling logs | (off) |
| `-Compare` | Path to golden metrics file for validation | (none) |
| `-ShowLog` | Display full log file after test completes | (off) |
| `-Json` | Output results as JSON instead of human-readable format | (off) |

### **Workflow Steps**

The script automates:

1. **Prerequisites Check**
   - Verify Python, pytest, scripts present
   - Check test files and golden metrics exist

2. **Deployment** (unless `-SkipDeploy`)
   - Runs `copy_to_kicad.ps1` to sync code to plugin folder

3. **Testing**
   - Smoke test: `pytest tests/regression/test_smoke.py::TestSmokeRouting::test_smoke_routing_pipeline`
   - Backplane test: `pytest tests/regression/test_backplane.py::TestHeadlessRouting::test_headless_routing_pipeline`

4. **Log Analysis**
   - Runs `scripts/analyze_log.py` on `logs/latest.log`
   - If `-Compare` specified, validates against golden metrics

5. **Results**
   - Clear pass/fail status with exit codes

### **Exit Codes**

| Code | Meaning | Action |
|------|---------|--------|
| **0** | Success (routing completed, validations passed) | ✅ Safe to commit |
| **1** | Routing failed (nets not routed, convergence failed) | ❌ Fix the bug |
| **2** | Performance regression (soft warnings) | ⚠️ Investigate regression |
| **3** | Script/environment error (prerequisites missing) | 🔧 Fix environment |

### **Example Workflows**

**Standard optimization cycle:**
```powershell
# 1. Edit code in unified_pathfinder.py
# 2. Run quick validation
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json

# 3. If passed, run full validation with profiling
.\scripts\optimize_and_validate.ps1 -ProfileMode -TestBoard backplane -Compare tests/regression/golden_metrics.json

# 4. If passed, commit changes
git commit -m "optimization: <description>"
```

**Debugging workflow:**
```powershell
# Run with full debug logs and display log after
.\scripts\optimize_and_validate.ps1 -ProfileMode -ShowLog

# Analyze specific sections
python scripts/analyze_log.py --log-file logs/latest.log
```

**CI/CD integration:**
```powershell
# In build pipeline
.\scripts\optimize_and_validate.ps1 -TestBoard smoke -Compare tests/regression/smoke_metrics.json -Json > results.json

# Parse results.json to determine pipeline status
```

---

## **Typical Optimization Workflow**

### **1. Initial Setup**

```powershell
# Ensure scripts are executable
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Verify prerequisites
python --version    # Ensure Python 3.10+
pytest --version    # Ensure pytest installed
```

### **2. Make Code Changes**

Edit source files (e.g., [orthoroute/algorithms/manhattan/unified_pathfinder.py](../orthoroute/algorithms/manhattan/unified_pathfinder.py))

### **3. Quick Validation**

```powershell
# Fast smoke test (~30s) to check for breakage
.\scripts\optimize_and_validate.ps1 -Compare tests/regression/smoke_metrics.json
```

**Exit code 0?** ✅ Proceed to step 4  
**Exit code 1/2?** ❌ Fix issues and repeat

### **4. Full Validation (Optional)**

```powershell
# Full backplane test with profiling
.\scripts\optimize_and_validate.ps1 -TestBoard backplane -ProfileMode -Compare tests/regression/golden_metrics.json
```

### **5. Analyze Results**

```powershell
# Detailed analysis with profiling breakdown
python scripts/analyze_log.py --compare tests/regression/golden_metrics.json
```

Review profiling data to identify new bottlenecks.

### **6. Commit or Rollback**

**If validation passed:**
```powershell
git add -A
git commit -m "optimization: <description of change>"
```

**If validation failed:**
```powershell
git restore .
# Or review specific issues and retry
```

### **7. Document Baseline (for significant improvements)**

Create new baseline doc in `docs/optimization/`:
```powershell
# Use template
cp docs/optimization/baseline_template.md docs/optimization/optimization_baseline_$(Get-Date -Format 'yyyy-MM-dd').md

# Fill in metrics from analyze_log.py output
```

---

## **Golden Metrics Files**

| File | Test Board | Purpose |
|------|------------|---------|
| [tests/regression/golden_metrics.json](../tests/regression/golden_metrics.json) | TestBackplane (512 nets) | Full golden standard for production performance |
| [tests/regression/smoke_metrics.json](../tests/regression/smoke_metrics.json) | Smoke (100 nets) | Fast validation checkpoint for quick iterations |

**Structure example:**
```json
{
  "nets_routed": 100,
  "total_nets": 100,
  "gpu": {
    "converged": true,
    "iterations_max": 20,
    "total_time_s_max": 60,
    "overuse_final_max": 0,
    "barrel_conflicts_max": 50
  }
}
```

---

## **Troubleshooting**

### **Script not found error**

```
python: can't open file 'scripts/analyze_log.py'
```

**Fix:** Run scripts from repo root:
```powershell
cd c:\Users\RWache\OneDrive - Rockwell Automation, Inc\Simulation tools\GitHub\OrthoRoute
python scripts/analyze_log.py
```

### **Log file not found**

```
Error: Log file not found: logs/latest.log
```

**Fix:** Run a routing test first:
```powershell
pytest tests/regression/test_smoke.py -v
# Then analyze
python scripts/analyze_log.py
```

### **Golden comparison shows all WARN/FAIL**

**Possible causes:**
1. Using wrong golden file (GPU metrics vs CPU metrics)
2. Golden thresholds outdated
3. Actual performance regression

**Investigate:**
```powershell
# Check actual metrics
python scripts/analyze_log.py --json

# Compare manually with golden file
cat tests/regression/smoke_metrics.json
```

### **pytest not found**

```
pytest: The term 'pytest' is not recognized
```

**Fix:**
```powershell
pip install -r requirements.txt
```

---

## **See Also**

- [docs/optimization/optimization_workflow.md](../docs/optimization/optimization_workflow.md) — Comprehensive optimization workflow guide
- [docs/optimization/OPTIMIZATION_QUICK_REF.md](../docs/optimization/OPTIMIZATION_QUICK_REF.md) — Quick reference for optimization
- [docs/optimization/README.md](../docs/optimization/README.md) — Optimization baselines and history
- [tests/run_golden_regression.md](../tests/run_golden_regression.md) — Golden regression test documentation
- [tests/README.md](../tests/README.md) — Test suite overview
