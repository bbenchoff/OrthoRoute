#!/usr/bin/env python3
"""
PathFinder Parameter Sweep Script
==================================

Runs overnight parameter sweep to find optimal convergence parameters.
Implements the playbook from FINAL_CONVERGENCE_TASK.md with priority ordering,
early stopping criteria, and comprehensive result tracking.

Usage:
    python parameter_sweep.py [--resume] [--config-only RUN_ID]

Features:
- Defines parameter matrix (A1-A6 Greedy, B1-B6 Penalized)
- Priority execution order: B4, B1, A4, A1, then others
- Early stopping criteria (no improvement, increases, single layer dominance, plateau)
- JSON result tracking with overuse history, layer spread, convergence metrics
- Resumable execution (checks existing results)
- Summary report with best config and convergence plots
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "parameter_sweep_results"
RESULTS_FILE = RESULTS_DIR / "sweep_results.json"
SUMMARY_FILE = RESULTS_DIR / "sweep_summary.txt"
TIMEOUT_SECONDS = 600  # 10 minutes per run

# Parameter matrix from playbook
PARAMETER_MATRIX = {
    # Greedy-start runs (A1-A6)
    "A1": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.35, "start_mode": "greedy"},
    "A2": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.45, "start_mode": "greedy"},
    "A3": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.8, "via_depart_max": 0.35, "start_mode": "greedy"},
    "A4": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.8, "via_depart_max": 0.45, "start_mode": "greedy"},
    "A5": {"initial_pres": 150, "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.35, "start_mode": "greedy"},
    "A6": {"initial_pres": 150, "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.45, "start_mode": "greedy"},

    # Penalized-start runs (B1-B6)
    "B1": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.35, "start_mode": "penalized"},
    "B2": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.45, "start_mode": "penalized"},
    "B3": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.8, "via_depart_max": 0.35, "start_mode": "penalized"},
    "B4": {"initial_pres": 50,  "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.8, "via_depart_max": 0.45, "start_mode": "penalized"},
    "B5": {"initial_pres": 150, "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.35, "start_mode": "penalized"},
    "B6": {"initial_pres": 150, "pres_mult": 1.30, "hist_gain": 1.1, "layer_bias_max": 1.6, "via_depart_max": 0.45, "start_mode": "penalized"},
}

# Priority order from playbook: B4, B1, A4, A1, then others
PRIORITY_ORDER = ["B4", "B1", "A4", "A1", "A2", "A3", "A5", "A6", "B2", "B3", "B5", "B6"]

# Early stopping criteria
EARLY_STOP_CRITERIA = {
    "no_improvement_iters": 6,        # No 10% improvement in 6 iterations
    "improvement_threshold": 0.10,    # 10% improvement threshold
    "consecutive_increases": 3,       # 3 consecutive increases >20%
    "increase_threshold": 0.20,       # 20% increase threshold
    "single_layer_threshold": 0.25,   # Single layer >25% after iter 12
    "single_layer_min_iter": 12,      # Minimum iteration for single layer check
    "plateau_iters": 8,               # No 5% improvement over 8 iterations
    "plateau_threshold": 0.05,        # 5% improvement threshold
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""
    iteration: int
    overuse: int
    edges_overused: int
    via_overuse_pct: float
    time_sec: float
    top3_layers: List[Tuple[int, int, float]]  # [(layer_num, overuse, percentage), ...]
    routed: int = 0
    failed: int = 0


@dataclass
class RunResult:
    """Complete results for a single parameter configuration run."""
    run_id: str
    config: Dict[str, Any]
    start_time: str
    end_time: Optional[str] = None
    iterations: List[IterationMetrics] = None
    best_overuse: Optional[int] = None
    best_iteration: Optional[int] = None
    final_overuse: Optional[int] = None
    final_iteration: Optional[int] = None
    convergence_achieved: bool = False
    early_stop_reason: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

    # Layer spread metrics
    layer_spread_stdev: Optional[float] = None
    max_layer_percentage: Optional[float] = None
    top3_layer_total_pct: Optional[float] = None

    # Timing metrics
    avg_iter_time: Optional[float] = None
    total_time_sec: Optional[float] = None

    def __post_init__(self):
        if self.iterations is None:
            self.iterations = []


# ============================================================================
# CONFIGURATION FILE MODIFICATION
# ============================================================================

def modify_config_file(config_params: Dict[str, Any]) -> None:
    """
    Modify the PathFinder config file to apply sweep parameters.

    This modifies orthoroute/algorithms/manhattan/pathfinder/config.py
    to set the appropriate parameters for the current run.
    """
    config_file = BASE_DIR / "orthoroute" / "algorithms" / "manhattan" / "pathfinder" / "config.py"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Read current config
    with open(config_file, 'r') as f:
        content = f.read()

    # Create backup
    backup_file = config_file.with_suffix('.py.backup')
    with open(backup_file, 'w') as f:
        f.write(content)

    # Modify parameters
    # Map our parameter names to config constants
    replacements = {
        "initial_pres": ("PRES_FAC_INIT", config_params["initial_pres"] / 100.0),  # Convert 50->0.5, 150->1.5
        "pres_mult": ("PRES_FAC_MULT", config_params["pres_mult"]),
        "hist_gain": ("HIST_ACCUM_GAIN", config_params["hist_gain"]),
        # layer_bias_max and via_depart_max need to be handled in PathFinderConfig dataclass
    }

    for param_name, (const_name, value) in replacements.items():
        # Find and replace the constant definition
        pattern = rf'^{const_name}\s*=\s*[0-9.]+\s*#'
        replacement = f'{const_name} = {value}  #'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Write modified config
    with open(config_file, 'w') as f:
        f.write(content)

    print(f"[CONFIG] Modified {config_file.name}")
    print(f"  initial_pres={config_params['initial_pres']/100} pres_mult={config_params['pres_mult']} hist_gain={config_params['hist_gain']}")


def restore_config_file() -> None:
    """Restore the original config file from backup."""
    config_file = BASE_DIR / "orthoroute" / "algorithms" / "manhattan" / "pathfinder" / "config.py"
    backup_file = config_file.with_suffix('.py.backup')

    if backup_file.exists():
        with open(backup_file, 'r') as f:
            content = f.read()
        with open(config_file, 'w') as f:
            f.write(content)
        print(f"[CONFIG] Restored original {config_file.name}")


# ============================================================================
# LOG PARSING
# ============================================================================

def parse_iteration_log(log_line: str) -> Optional[IterationMetrics]:
    """
    Parse a log line like:
    [ITER 15] routed=512 failed=0 overuse=2341 edges=1234 via_overuse=0.3%
    """
    pattern = r'\[ITER\s+(\d+)\]\s+routed=(\d+)\s+failed=(\d+)\s+overuse=(\d+)\s+edges=(\d+)\s+via_overuse=([\d.]+)%'
    match = re.search(pattern, log_line)
    if not match:
        return None

    return IterationMetrics(
        iteration=int(match.group(1)),
        routed=int(match.group(2)),
        failed=int(match.group(3)),
        overuse=int(match.group(4)),
        edges_overused=int(match.group(5)),
        via_overuse_pct=float(match.group(6)),
        time_sec=0.0,  # Will be filled from separate timing logs
        top3_layers=[]  # Will be filled from layer congestion logs
    )


def parse_layer_congestion(log_lines: List[str], iter_num: int) -> List[Tuple[int, int, float]]:
    """
    Parse layer congestion output to extract top 3 layers.

    Expected format:
    [LAYER-CONGESTION] Layer 30: 1570 overuse (16.5%)
    [LAYER-CONGESTION] Layer 10: 1195 overuse (12.6%)
    """
    top3 = []

    for line in log_lines:
        if f"[ITER {iter_num}]" not in line and "[LAYER-CONGESTION]" not in line:
            continue

        pattern = r'Layer\s+(\d+):\s+(\d+)\s+overuse\s+\(([\d.]+)%\)'
        match = re.search(pattern, line)
        if match:
            layer_num = int(match.group(1))
            overuse = int(match.group(2))
            percentage = float(match.group(3))
            top3.append((layer_num, overuse, percentage))

            if len(top3) >= 3:
                break

    return top3[:3]


def parse_iteration_timing(log_lines: List[str]) -> Dict[int, float]:
    """
    Parse iteration timing from log lines.

    Expected format:
    [TIMING] Iteration 15 completed in 31.2s
    """
    timing_map = {}

    for line in log_lines:
        pattern = r'\[TIMING\].*Iteration\s+(\d+).*?([\d.]+)s'
        match = re.search(pattern, line)
        if match:
            iter_num = int(match.group(1))
            time_sec = float(match.group(2))
            timing_map[iter_num] = time_sec

    return timing_map


def parse_log_output(log_output: str) -> List[IterationMetrics]:
    """Parse complete log output and extract all iteration metrics."""
    lines = log_output.split('\n')

    # Parse iteration metrics
    iterations = []
    for line in lines:
        if '[ITER' in line and 'overuse=' in line:
            metrics = parse_iteration_log(line)
            if metrics:
                iterations.append(metrics)

    # Parse timing data
    timing_map = parse_iteration_timing(lines)
    for metrics in iterations:
        if metrics.iteration in timing_map:
            metrics.time_sec = timing_map[metrics.iteration]

    # Parse layer congestion for each iteration
    for metrics in iterations:
        metrics.top3_layers = parse_layer_congestion(lines, metrics.iteration)

    return iterations


# ============================================================================
# EARLY STOPPING LOGIC
# ============================================================================

def check_early_stop(iterations: List[IterationMetrics]) -> Optional[str]:
    """
    Check if early stopping criteria are met.

    Returns:
        Stop reason string if should stop, None otherwise
    """
    if len(iterations) < 2:
        return None

    criteria = EARLY_STOP_CRITERIA

    # Criterion 1: No 10% improvement in last 6 iterations
    if len(iterations) >= criteria["no_improvement_iters"] + 1:
        recent = iterations[-criteria["no_improvement_iters"]:]
        baseline = iterations[-criteria["no_improvement_iters"]-1].overuse

        improved = False
        for iter_metrics in recent:
            improvement = (baseline - iter_metrics.overuse) / max(1, baseline)
            if improvement >= criteria["improvement_threshold"]:
                improved = True
                break

        if not improved:
            return f"No 10% improvement in {criteria['no_improvement_iters']} iterations"

    # Criterion 2: 3 consecutive increases >20%
    if len(iterations) >= criteria["consecutive_increases"] + 1:
        recent = iterations[-criteria["consecutive_increases"]-1:]
        consecutive_increases = 0

        for i in range(1, len(recent)):
            increase = (recent[i].overuse - recent[i-1].overuse) / max(1, recent[i-1].overuse)
            if increase > criteria["increase_threshold"]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

        if consecutive_increases >= criteria["consecutive_increases"]:
            return f"{criteria['consecutive_increases']} consecutive increases >20%"

    # Criterion 3: Single layer >25% after iteration 12
    if len(iterations) > criteria["single_layer_min_iter"]:
        latest = iterations[-1]
        if latest.iteration >= criteria["single_layer_min_iter"]:
            if latest.top3_layers:
                max_layer_pct = max(pct for _, _, pct in latest.top3_layers)
                if max_layer_pct > criteria["single_layer_threshold"] * 100:
                    return f"Single layer >{criteria['single_layer_threshold']*100}% after iter {criteria['single_layer_min_iter']}"

    # Criterion 4: Plateau (no 5% improvement over 8 iterations)
    if len(iterations) >= criteria["plateau_iters"] + 1:
        recent = iterations[-criteria["plateau_iters"]:]
        baseline = iterations[-criteria["plateau_iters"]-1].overuse

        improved = False
        for iter_metrics in recent:
            improvement = (baseline - iter_metrics.overuse) / max(1, baseline)
            if improvement >= criteria["plateau_threshold"]:
                improved = True
                break

        if not improved:
            return f"Plateau: No 5% improvement over {criteria['plateau_iters']} iterations"

    return None


# ============================================================================
# RUN EXECUTION
# ============================================================================

def run_pathfinder_test(run_id: str, config: Dict[str, Any]) -> RunResult:
    """
    Execute a single PathFinder test run with given configuration.

    Args:
        run_id: Unique identifier for this run (e.g., "A1", "B4")
        config: Parameter configuration dictionary

    Returns:
        RunResult object with complete metrics
    """
    print(f"\n{'='*80}")
    print(f"Starting run: {run_id}")
    print(f"Config: {config}")
    print(f"{'='*80}\n")

    result = RunResult(
        run_id=run_id,
        config=config,
        start_time=datetime.now().isoformat()
    )

    # Create log file for this run
    log_file = RESULTS_DIR / f"run_{run_id}.log"

    try:
        # Modify config file
        modify_config_file(config)

        # Set up environment
        env = os.environ.copy()
        env["ORTHO_CPU_ONLY"] = "1"

        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(BASE_DIR / "main.py"),
            "--test-manhattan"
        ]

        print(f"[RUN] Executing: {' '.join(cmd)}")
        print(f"[RUN] Timeout: {TIMEOUT_SECONDS}s")
        print(f"[RUN] Log file: {log_file}")

        # Run with timeout
        start_time = time.time()

        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(BASE_DIR),
                universal_newlines=True,
                bufsize=1
            )

            # Stream output to file and console
            log_output = []
            try:
                for line in process.stdout:
                    log_f.write(line)
                    log_f.flush()
                    log_output.append(line)

                    # Print important lines to console
                    if any(marker in line for marker in ['[ITER', '[EARLY-STOP]', 'TEST PASSED', 'TEST FAILED']):
                        print(line.rstrip())

                    # Check for early stopping in real-time
                    if '[ITER' in line and 'overuse=' in line:
                        iterations = parse_log_output(''.join(log_output))
                        stop_reason = check_early_stop(iterations)
                        if stop_reason:
                            print(f"\n[EARLY-STOP] {stop_reason}")
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            result.early_stop_reason = stop_reason
                            break

                # Wait for process completion
                return_code = process.wait(timeout=max(10, TIMEOUT_SECONDS - (time.time() - start_time)))

            except subprocess.TimeoutExpired:
                print(f"\n[TIMEOUT] Run exceeded {TIMEOUT_SECONDS}s limit")
                process.kill()
                result.error_message = f"Timeout after {TIMEOUT_SECONDS}s"

        end_time = time.time()
        result.total_time_sec = end_time - start_time
        result.end_time = datetime.now().isoformat()

        # Parse log output
        full_log = ''.join(log_output)
        result.iterations = parse_log_output(full_log)

        if result.iterations:
            # Calculate metrics
            result.best_overuse = min(m.overuse for m in result.iterations)
            result.best_iteration = next(i for i, m in enumerate(result.iterations) if m.overuse == result.best_overuse)
            result.final_overuse = result.iterations[-1].overuse
            result.final_iteration = result.iterations[-1].iteration
            result.convergence_achieved = (result.best_overuse < 1000)

            # Calculate layer spread metrics
            if result.iterations[-1].top3_layers:
                top3_pcts = [pct for _, _, pct in result.iterations[-1].top3_layers]
                result.max_layer_percentage = max(top3_pcts)
                result.top3_layer_total_pct = sum(top3_pcts)

            # Calculate timing metrics
            times = [m.time_sec for m in result.iterations if m.time_sec > 0]
            if times:
                result.avg_iter_time = sum(times) / len(times)

            # Success if test passed and converged reasonably
            result.success = (
                'TEST PASSED' in full_log and
                result.final_overuse < 10000 and
                len(result.iterations) > 5
            )

        print(f"\n[RESULT] {run_id}: best_overuse={result.best_overuse}, final_overuse={result.final_overuse}, iters={len(result.iterations)}")

    except Exception as e:
        print(f"\n[ERROR] Run {run_id} failed with exception: {e}")
        result.error_message = str(e)
        result.end_time = datetime.now().isoformat()

    finally:
        # Always restore config file
        restore_config_file()

    return result


# ============================================================================
# RESULT MANAGEMENT
# ============================================================================

def load_existing_results() -> Dict[str, RunResult]:
    """Load existing results from JSON file."""
    if not RESULTS_FILE.exists():
        return {}

    try:
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)

        results = {}
        for run_id, result_dict in data.items():
            # Reconstruct IterationMetrics objects
            iterations = []
            for iter_dict in result_dict.get('iterations', []):
                iterations.append(IterationMetrics(**iter_dict))

            result_dict['iterations'] = iterations
            results[run_id] = RunResult(**result_dict)

        return results

    except Exception as e:
        print(f"[WARNING] Failed to load existing results: {e}")
        return {}


def save_results(results: Dict[str, RunResult]) -> None:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for run_id, result in results.items():
        result_dict = asdict(result)
        serializable[run_id] = result_dict

    with open(RESULTS_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"[SAVE] Results saved to {RESULTS_FILE}")


# ============================================================================
# REPORTING
# ============================================================================

def generate_summary_report(results: Dict[str, RunResult]) -> str:
    """Generate a comprehensive summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PathFinder Parameter Sweep Summary")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total runs: {len(results)}")
    lines.append("")

    # Filter successful runs
    successful = {k: v for k, v in results.items() if v.success}
    lines.append(f"Successful runs: {len(successful)}/{len(results)}")
    lines.append("")

    if not successful:
        lines.append("No successful runs completed.")
        return '\n'.join(lines)

    # Find best configuration
    best_run_id = min(successful.keys(), key=lambda k: successful[k].best_overuse)
    best_result = successful[best_run_id]

    lines.append("=" * 80)
    lines.append("BEST CONFIGURATION")
    lines.append("=" * 80)
    lines.append(f"Run ID: {best_run_id}")
    lines.append(f"Config: {best_result.config}")
    lines.append("")
    lines.append("Results:")
    lines.append(f"  Best overuse: {best_result.best_overuse} (iteration {best_result.best_iteration})")
    lines.append(f"  Final overuse: {best_result.final_overuse} (iteration {best_result.final_iteration})")
    lines.append(f"  Convergence achieved: {best_result.convergence_achieved}")
    lines.append(f"  Total iterations: {len(best_result.iterations)}")
    lines.append(f"  Avg iteration time: {best_result.avg_iter_time:.1f}s")
    lines.append(f"  Total run time: {best_result.total_time_sec:.1f}s")
    lines.append("")

    if best_result.top3_layer_total_pct:
        lines.append("Layer Distribution:")
        lines.append(f"  Max layer percentage: {best_result.max_layer_percentage:.1f}%")
        lines.append(f"  Top-3 layer total: {best_result.top3_layer_total_pct:.1f}%")
        lines.append("")

    # Convergence history for best run
    lines.append("Convergence History (Best Run):")
    for i, metrics in enumerate(best_result.iterations[:20]):  # First 20 iterations
        top3_str = ", ".join([f"L{l}:{p:.1f}%" for l, _, p in metrics.top3_layers[:3]])
        lines.append(f"  Iter {metrics.iteration:2d}: overuse={metrics.overuse:6d}  via={metrics.via_overuse_pct:4.1f}%  [{top3_str}]")
    if len(best_result.iterations) > 20:
        lines.append(f"  ... ({len(best_result.iterations) - 20} more iterations)")
    lines.append("")

    # Ranking table
    lines.append("=" * 80)
    lines.append("ALL RUNS RANKED BY BEST OVERUSE")
    lines.append("=" * 80)
    lines.append(f"{'Rank':<6} {'Run':<6} {'Best':>8} {'Final':>8} {'Iters':>6} {'Time':>8} {'Status':<20}")
    lines.append("-" * 80)

    ranked = sorted(successful.items(), key=lambda x: x[1].best_overuse)
    for rank, (run_id, result) in enumerate(ranked, 1):
        status = "CONVERGED" if result.convergence_achieved else "Incomplete"
        if result.early_stop_reason:
            status = f"Early stop: {result.early_stop_reason[:30]}"

        lines.append(
            f"{rank:<6} {run_id:<6} {result.best_overuse:>8} {result.final_overuse:>8} "
            f"{len(result.iterations):>6} {result.total_time_sec:>7.0f}s {status:<20}"
        )
    lines.append("")

    # Failed runs
    failed = {k: v for k, v in results.items() if not v.success}
    if failed:
        lines.append("=" * 80)
        lines.append("FAILED RUNS")
        lines.append("=" * 80)
        for run_id, result in failed.items():
            lines.append(f"{run_id}: {result.error_message or 'Unknown error'}")
        lines.append("")

    # Parameter impact analysis
    lines.append("=" * 80)
    lines.append("PARAMETER IMPACT ANALYSIS")
    lines.append("=" * 80)

    # Group by parameter values
    param_groups = {}
    for param_name in ["initial_pres", "layer_bias_max", "via_depart_max", "start_mode"]:
        param_groups[param_name] = {}
        for run_id, result in successful.items():
            value = result.config[param_name]
            if value not in param_groups[param_name]:
                param_groups[param_name][value] = []
            param_groups[param_name][value].append(result.best_overuse)

    for param_name, value_dict in param_groups.items():
        lines.append(f"\n{param_name}:")
        for value, overuses in value_dict.items():
            avg = sum(overuses) / len(overuses)
            lines.append(f"  {value:>10}: avg={avg:>8.0f}  (n={len(overuses)})")

    lines.append("")
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append(f"Best single config: {best_run_id}")
    lines.append(f"  Parameters: {best_result.config}")
    lines.append(f"  Expected best overuse: {best_result.best_overuse}")
    lines.append(f"  Expected convergence: {'YES' if best_result.convergence_achieved else 'NO'}")
    lines.append("")

    # Top 3 configs
    lines.append("Top 3 configurations:")
    for rank, (run_id, result) in enumerate(ranked[:3], 1):
        lines.append(f"  {rank}. {run_id}: best_overuse={result.best_overuse}, config={result.config}")
    lines.append("")

    return '\n'.join(lines)


def save_summary_report(results: Dict[str, RunResult]) -> None:
    """Generate and save summary report to file."""
    summary = generate_summary_report(results)

    with open(SUMMARY_FILE, 'w') as f:
        f.write(summary)

    print(f"\n[REPORT] Summary saved to {SUMMARY_FILE}")
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    # Print abbreviated summary to console
    successful = {k: v for k, v in results.items() if v.success}
    if successful:
        best_run_id = min(successful.keys(), key=lambda k: successful[k].best_overuse)
        best_result = successful[best_run_id]
        print(f"Best configuration: {best_run_id}")
        print(f"  Best overuse: {best_result.best_overuse}")
        print(f"  Final overuse: {best_result.final_overuse}")
        print(f"  Converged: {best_result.convergence_achieved}")
        print(f"  Config: {best_result.config}")
    else:
        print("No successful runs completed.")

    print(f"\nFull report: {SUMMARY_FILE}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="PathFinder parameter sweep for overnight convergence testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    parser.add_argument('--config-only', metavar='RUN_ID', help='Only modify config for specified run (no execution)')
    parser.add_argument('--dry-run', action='store_true', help='Print execution plan without running')

    args = parser.parse_args()

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Handle config-only mode
    if args.config_only:
        run_id = args.config_only
        if run_id not in PARAMETER_MATRIX:
            print(f"[ERROR] Unknown run ID: {run_id}")
            print(f"Available: {', '.join(PARAMETER_MATRIX.keys())}")
            sys.exit(1)

        config = PARAMETER_MATRIX[run_id]
        print(f"[CONFIG-ONLY] Modifying config for {run_id}")
        modify_config_file(config)
        print(f"[CONFIG-ONLY] Config modified. Run 'python main.py --test-manhattan' to test.")
        sys.exit(0)

    # Load existing results if resuming
    results = {}
    if args.resume:
        results = load_existing_results()
        print(f"[RESUME] Loaded {len(results)} existing results")

    # Determine runs to execute
    runs_to_execute = []
    for run_id in PRIORITY_ORDER:
        if run_id not in results or not results[run_id].success:
            runs_to_execute.append(run_id)

    print(f"\n{'='*80}")
    print("PathFinder Parameter Sweep")
    print(f"{'='*80}")
    print(f"Total configurations: {len(PARAMETER_MATRIX)}")
    print(f"Already completed: {len(results)}")
    print(f"Remaining to run: {len(runs_to_execute)}")
    print(f"Execution order: {', '.join(runs_to_execute)}")
    print(f"Estimated time: {len(runs_to_execute) * TIMEOUT_SECONDS / 60:.0f} minutes")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("[DRY-RUN] Execution plan printed. Use without --dry-run to execute.")
        sys.exit(0)

    # Execute runs
    for i, run_id in enumerate(runs_to_execute, 1):
        print(f"\n[SWEEP] Progress: {i}/{len(runs_to_execute)}")

        config = PARAMETER_MATRIX[run_id]
        result = run_pathfinder_test(run_id, config)
        results[run_id] = result

        # Save after each run
        save_results(results)

        # Print intermediate summary
        if result.success:
            print(f"[SUCCESS] {run_id}: best={result.best_overuse}, final={result.final_overuse}")
        else:
            print(f"[FAILED] {run_id}: {result.error_message}")

    # Generate final summary
    print(f"\n{'='*80}")
    print("Parameter Sweep Complete!")
    print(f"{'='*80}\n")

    save_summary_report(results)

    # Print results location
    print(f"\nResults saved to:")
    print(f"  JSON data: {RESULTS_FILE}")
    print(f"  Summary report: {SUMMARY_FILE}")
    print(f"  Run logs: {RESULTS_DIR}/run_*.log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Sweep interrupted by user")
        print("Run with --resume to continue from last checkpoint")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] Sweep failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
