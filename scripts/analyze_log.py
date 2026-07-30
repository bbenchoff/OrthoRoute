#!/usr/bin/env python3
"""
OrthoRoute Log Analyzer

Standalone log parsing tool for OrthoRoute routing logs.
Extracts routing metrics, convergence status, and performance data.

Usage:
    python scripts/analyze_log.py                           # Parse logs/latest.log
    python scripts/analyze_log.py --log-file path/to/log    # Parse specific log
    python scripts/analyze_log.py --json                    # Output JSON
    python scripts/analyze_log.py --compare golden.json     # Compare with thresholds
    python scripts/analyze_log.py --compare golden.json --json  # JSON comparison

Examples:
    # Quick analysis of latest run
    python scripts/analyze_log.py

    # Compare against golden metrics
    python scripts/analyze_log.py --compare tests/regression/golden_metrics.json

    # Export metrics as JSON for automation
    python scripts/analyze_log.py --json > metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# =============================================================================
# LOG PARSING FUNCTIONS
# =============================================================================

def parse_routing_result(text: str) -> dict[str, Any] | None:
    """
    Extract comprehensive routing metrics from an OrthoRoute log.

    Parses three layers of evidence:
      1. [ITER N] lines          — per-iteration timing + overuse
      2. End-of-run summary      — ROUTING COMPLETE / [CLEAN] / [FINAL] lines
      3. Write-back confirmation — "Applied N tracks and N vias to KiCad"

    Returns a routing_result-compatible dict, or None if the log has no
    routing completion marker.

    Args:
        text: Log file content as string

    Returns:
        Dictionary with routing metrics or None if incomplete
    """
    if "ROUTING COMPLETE" not in text:
        return None

    # --- Parse [ITER N] lines for per-iteration data ---
    # Example: [ITER  67] nets=512/512  ✓ CONVERGED  edges=0  via_overuse=0%  barrel=379  iter=4.0s  total=761.2s
    iter_pattern = re.compile(
        r"\[ITER\s+(\d+)\].*?nets=(\d+)/(\d+).*?edges=(\d+).*?barrel=(\d+).*?iter=([0-9.]+)s.*?total=([0-9.]+)s"
    )
    iter_rows = []
    final_iter = None
    for ln in text.splitlines():
        m = iter_pattern.search(ln)
        if m:
            row = {
                "iter": int(m.group(1)),
                "nets_routed": int(m.group(2)),
                "total_nets": int(m.group(3)),
                "overuse_edges": int(m.group(4)),
                "barrel": int(m.group(5)),
                "iter_time_s": float(m.group(6)),
                "total_time_s": float(m.group(7)),
                "converged": "CONVERGED" in ln,
            }
            iter_rows.append(row)
            if "CONVERGED" in ln:
                final_iter = row

    if not iter_rows:
        return None

    last = final_iter or iter_rows[-1]

    # --- Parse end-of-run summary block ---
    clean_zero_overuse = bool(re.search(r"\[CLEAN\] All nets routed with zero overuse", text))

    rc_m = re.search(r"ROUTING COMPLETE: All (\d+) nets routed successfully", text)
    routing_complete_nets = int(rc_m.group(1)) if rc_m else None

    final_m = re.search(r"\[FINAL\] Edge routing converged \((\d+) barrel conflicts remain", text)
    final_barrel_conflicts = int(final_m.group(1)) if final_m else last["barrel"]

    hard_converged = clean_zero_overuse or (rc_m is not None)

    # --- Parse writeback confirmation ---
    wb_m = re.search(r"Applied (\d+) tracks and (\d+) vias", text)
    tracks_out = int(wb_m.group(1)) if wb_m else None
    vias_out = int(wb_m.group(2)) if wb_m else None

    # Build summary message
    if rc_m:
        message = f"ROUTING COMPLETE: All {routing_complete_nets} nets routed with zero overuse"
        if final_barrel_conflicts:
            message += f" ({final_barrel_conflicts} barrel conflicts — acceptable)"
    else:
        message = f"Parsed from log: {last['nets_routed']}/{last['total_nets']} nets routed"

    return {
        "success": True,
        "converged": hard_converged or last.get("converged", False),
        "nets_routed": routing_complete_nets or last["nets_routed"],
        "total_nets": last["total_nets"],
        "iterations": last["iter"],
        "total_time_s": last["total_time_s"],
        "iteration_metrics": [
            {"iter": r["iter"], "iter_time_s": r["iter_time_s"]} for r in iter_rows
        ],
        "failed_nets": last["total_nets"] - (routing_complete_nets or last["nets_routed"]),
        "overuse_sum": 0 if clean_zero_overuse else last["overuse_edges"],
        "overuse_edges": 0 if clean_zero_overuse else last["overuse_edges"],
        "barrel_conflicts": final_barrel_conflicts,
        "tracks_written": tracks_out,
        "vias_written": vias_out,
        "clean_zero_overuse": clean_zero_overuse,
        "routing_complete_banner": rc_m is not None,
        "message": message,
        "_source": "log_parse",
    }


def parse_lattice_dimensions(text: str) -> dict[str, int] | None:
    """
    Extract lattice dimensions from log.

    Example log line:
        WARNING - Lattice: 106×234×18 = 446,472 nodes

    Args:
        text: Log file content

    Returns:
        Dict with cols, rows, layers, nodes or None if not found
    """
    pattern = r"Lattice:\s*(\d+)×(\d+)×(\d+)\s*=\s*([\d,]+)\s*nodes"
    m = re.search(pattern, text)
    if m:
        return {
            "cols": int(m.group(1)),
            "rows": int(m.group(2)),
            "layers": int(m.group(3)),
            "nodes": int(m.group(4).replace(",", "")),
        }
    return None


def detect_gpu_mode(text: str) -> bool:
    """
    Detect if GPU mode was used in this routing run.

    Args:
        text: Log file content

    Returns:
        True if GPU detected, False if CPU-only
    """
    # Look for GPU-specific markers in logs
    gpu_markers = [
        r"\[GPU\]",
        r"CUDA kernel",
        r"GPU mode: ON",
        r"CuPy available",
    ]
    for marker in gpu_markers:
        if re.search(marker, text, re.IGNORECASE):
            return True
    return False


def parse_profile_times(text: str) -> dict[str, list[float]]:
    """
    Extract [PROFILE] timing data from debug logs.

    Example log line:
        WARNING - [PROFILE] _build_owner_bitmap_for_fullgraph: 0.92ms

    Args:
        text: Log file content

    Returns:
        Dict mapping function names to list of timing measurements (ms)
    """
    pattern = r"\[PROFILE\]\s+(\w+):\s+([\d.]+)ms"
    profile_data = {}
    
    for m in re.finditer(pattern, text):
        func_name = m.group(1)
        time_ms = float(m.group(2))
        if func_name not in profile_data:
            profile_data[func_name] = []
        profile_data[func_name].append(time_ms)
    
    return profile_data


# =============================================================================
# COMPARISON LOGIC
# =============================================================================

def compare_with_golden(metrics: dict[str, Any], golden_path: Path) -> dict[str, Any]:
    """
    Compare routing metrics against golden thresholds.

    Args:
        metrics: Parsed routing metrics
        golden_path: Path to golden_metrics.json file

    Returns:
        Dictionary with comparison results
    """
    with golden_path.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    # Determine if GPU or CPU mode
    mode = "gpu"  # Default to GPU, could be auto-detected from log
    thresholds = golden.get(mode, golden)  # Fallback to root-level thresholds

    checks = []
    
    # Required metrics (hard failures)
    required_checks = [
        ("nets_routed", metrics.get("nets_routed"), golden.get("nets_routed"), "eq"),
        ("total_nets", metrics.get("total_nets"), golden.get("total_nets"), "eq"),
        ("converged", metrics.get("converged"), True, "eq"),
    ]
    
    for name, actual, expected, check_type in required_checks:
        if expected is None:
            continue
        if check_type == "eq":
            passed = actual == expected
            status = "PASS" if passed else "FAIL"
        else:
            passed = actual <= expected
            status = "PASS" if passed else "FAIL"
        
        checks.append({
            "metric": name,
            "actual": actual,
            "expected": expected,
            "status": status,
            "required": True,
        })
    
    # Performance metrics (soft warnings)
    perf_checks = [
        ("iterations", metrics.get("iterations"), thresholds.get("iterations_max"), "le"),
        ("total_time_s", metrics.get("total_time_s"), thresholds.get("total_time_s_max"), "le"),
        ("barrel_conflicts", metrics.get("barrel_conflicts"), thresholds.get("barrel_conflicts_max"), "le"),
    ]
    
    for name, actual, threshold, check_type in perf_checks:
        if threshold is None:
            continue
        passed = actual <= threshold if check_type == "le" else actual == threshold
        status = "PASS" if passed else "WARN"
        
        checks.append({
            "metric": name,
            "actual": actual,
            "threshold": threshold,
            "status": status,
            "required": False,
        })
    
    # Overall status: PASS if no FAIL, WARN if any WARN, FAIL otherwise
    has_fail = any(c["status"] == "FAIL" for c in checks)
    has_warn = any(c["status"] == "WARN" for c in checks)
    
    if has_fail:
        overall = "FAIL"
    elif has_warn:
        overall = "WARN"
    else:
        overall = "PASS"
    
    return {
        "overall_status": overall,
        "mode": mode,
        "checks": checks,
    }


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_human_readable(metrics: dict[str, Any], lattice: dict | None, 
                          gpu_mode: bool, profile_data: dict | None,
                          comparison: dict | None) -> str:
    """
    Format metrics as human-readable text.

    Args:
        metrics: Parsed routing metrics
        lattice: Lattice dimensions (optional)
        gpu_mode: GPU mode detected
        profile_data: Profiling data (optional)
        comparison: Comparison results (optional)

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("OrthoRoute Log Analysis")
    lines.append("=" * 80)
    
    # Basic metrics
    lines.append("")
    lines.append("ROUTING SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Nets routed:       {metrics.get('nets_routed')}/{metrics.get('total_nets')}")
    lines.append(f"  Converged:         {metrics.get('converged')}")
    lines.append(f"  Iterations:        {metrics.get('iterations')}")
    lines.append(f"  Total time:        {metrics.get('total_time_s'):.1f}s ({metrics.get('total_time_s')/60:.1f} min)")
    
    if metrics.get('iterations'):
        avg_iter = metrics.get('total_time_s') / metrics.get('iterations')
        lines.append(f"  Avg iteration:     {avg_iter:.1f}s")
    
    lines.append(f"  Overuse edges:     {metrics.get('overuse_edges', 0)}")
    lines.append(f"  Barrel conflicts:  {metrics.get('barrel_conflicts', 0)}")
    
    if metrics.get('tracks_written') is not None:
        lines.append(f"  Tracks written:    {metrics.get('tracks_written')}")
        lines.append(f"  Vias written:      {metrics.get('vias_written')}")
    
    lines.append(f"  GPU mode:          {'YES' if gpu_mode else 'NO (CPU-only)'}")
    
    # Lattice info
    if lattice:
        lines.append("")
        lines.append("LATTICE DIMENSIONS")
        lines.append("-" * 80)
        lines.append(f"  Grid:              {lattice['cols']}×{lattice['rows']}×{lattice['layers']}")
        lines.append(f"  Total nodes:       {lattice['nodes']:,}")
    
    # Profile data summary
    if profile_data:
        lines.append("")
        lines.append("PROFILING DATA (Top 10 by total time)")
        lines.append("-" * 80)
        
        # Calculate totals and sort
        profile_summary = []
        for func, times in profile_data.items():
            total_ms = sum(times)
            count = len(times)
            avg_ms = total_ms / count if count > 0 else 0
            profile_summary.append((func, total_ms, count, avg_ms))
        
        profile_summary.sort(key=lambda x: x[1], reverse=True)
        
        for func, total_ms, count, avg_ms in profile_summary[:10]:
            lines.append(f"  {func:40s} {total_ms/1000:8.1f}s  ({count:4d} calls, {avg_ms:6.1f}ms avg)")
    
    # Comparison results
    if comparison:
        lines.append("")
        lines.append("GOLDEN COMPARISON")
        lines.append("-" * 80)
        lines.append(f"  Overall status: {comparison['overall_status']}")
        lines.append(f"  Mode:           {comparison['mode'].upper()}")
        lines.append("")
        
        for check in comparison['checks']:
            metric = check['metric']
            actual = check['actual']
            status = check['status']
            
            if 'expected' in check:
                expected = check['expected']
                symbol = "✓" if status == "PASS" else "✗"
                lines.append(f"  {symbol} {metric:20s} actual={actual:8} expected={expected:8} => {status}")
            elif 'threshold' in check:
                threshold = check['threshold']
                symbol = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
                if isinstance(actual, float):
                    lines.append(f"  {symbol} {metric:20s} actual={actual:8.1f} threshold={threshold:8.1f} => {status}")
                else:
                    lines.append(f"  {symbol} {metric:20s} actual={actual:8} threshold={threshold:8} => {status}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def format_json_output(metrics: dict[str, Any], lattice: dict | None,
                       gpu_mode: bool, profile_data: dict | None,
                       comparison: dict | None) -> str:
    """
    Format metrics as JSON.

    Args:
        metrics: Parsed routing metrics
        lattice: Lattice dimensions (optional)
        gpu_mode: GPU mode detected
        profile_data: Profiling data (optional)
        comparison: Comparison results (optional)

    Returns:
        JSON string
    """
    output = {
        "routing_summary": metrics,
        "lattice": lattice,
        "gpu_mode": gpu_mode,
    }
    
    if profile_data:
        # Summarize profile data
        profile_summary = {}
        for func, times in profile_data.items():
            profile_summary[func] = {
                "total_ms": sum(times),
                "count": len(times),
                "avg_ms": sum(times) / len(times) if times else 0,
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            }
        output["profiling"] = profile_summary
    
    if comparison:
        output["comparison"] = comparison
    
    return json.dumps(output, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze OrthoRoute routing logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: logs/latest.log)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable format",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare against golden metrics file (e.g., tests/regression/golden_metrics.json)",
    )
    
    args = parser.parse_args()
    
    # Determine log file path
    if args.log_file:
        log_file = args.log_file
    else:
        # Default to logs/latest.log relative to repo root
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        log_file = repo_root / "logs" / "latest.log"
    
    # Read log file
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}", file=sys.stderr)
        return 1
    
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return 1
    
    # Parse metrics
    metrics = parse_routing_result(text)
    if metrics is None:
        print("Error: Log does not contain routing completion data", file=sys.stderr)
        print("  (Looking for 'ROUTING COMPLETE' banner)", file=sys.stderr)
        return 1
    
    lattice = parse_lattice_dimensions(text)
    gpu_mode = detect_gpu_mode(text)
    profile_data = parse_profile_times(text)
    
    # Optional comparison
    comparison = None
    if args.compare:
        if not args.compare.exists():
            print(f"Error: Golden metrics file not found: {args.compare}", file=sys.stderr)
            return 1
        comparison = compare_with_golden(metrics, args.compare)
    
    # Output
    if args.json:
        output = format_json_output(metrics, lattice, gpu_mode, profile_data, comparison)
        print(output)
    else:
        output = format_human_readable(metrics, lattice, gpu_mode, profile_data, comparison)
        print(output)
    
    # Exit code based on comparison (if performed)
    if comparison:
        if comparison["overall_status"] == "FAIL":
            return 1
        elif comparison["overall_status"] == "WARN":
            return 0  # Warnings don't fail the script
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
