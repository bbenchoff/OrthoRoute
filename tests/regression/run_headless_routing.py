#!/usr/bin/env python3
"""
Headless regression runner for TestBackplane.

Runs full routing using the file parser and validates completion by scanning
the generated log for the ROUTING COMPLETE banner before a timeout threshold.

Timeout policy:
- Base timeout is the golden GPU max total-time threshold from
  tests/regression/golden_metrics.json (gpu.total_time_s_max).
- A +10% buffer is applied.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

# Enable detailed logs before imports.
os.environ["ORTHO_DEBUG"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orthoroute.algorithms.manhattan.unified_pathfinder import (  # noqa: E402
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser  # noqa: E402
from orthoroute.shared.utils.logging_utils import init_logging  # noqa: E402


def _load_golden_metrics() -> dict:
    golden_path = REPO_ROOT / "tests" / "regression" / "golden_metrics.json"
    with golden_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compute_timeout_seconds(golden_metrics: dict) -> int:
    gpu = golden_metrics.get("gpu", {})
    base_timeout = float(gpu.get("total_time_s_max", golden_metrics.get("total_time_s_max", 1800)))
    return int(math.ceil(base_timeout * 1.10))


def _find_latest_log_file() -> Path | None:
    logs_dir = REPO_ROOT / "logs"
    if not logs_dir.exists():
        return None
    logs = list(logs_dir.glob("*.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _parse_log_completion_metrics(log_text: str) -> dict:
    has_banner = "ROUTING COMPLETE" in log_text
    nets_match = re.search(r"Nets routed:\s*(\d+)/(\d+)", log_text)
    converged_match = re.search(r"Converged:\s*(True|False)", log_text)

    iter_pattern = re.compile(
        r"\[ITER\s+(\d+)\].*?nets=(\d+)/(\d+).*?(?:✓\s*CONVERGED|overuse=\d+).*?total=([0-9.]+)s"
    )
    iter_rows = [
        {
            "iter": int(m.group(1)),
            "nets_routed": int(m.group(2)),
            "total_nets": int(m.group(3)),
            "total_time_s": float(m.group(4)),
        }
        for m in iter_pattern.finditer(log_text)
    ]

    last_iter = iter_rows[-1] if iter_rows else None
    return {
        "has_routing_complete": has_banner,
        "nets_routed": int(nets_match.group(1)) if nets_match else (last_iter["nets_routed"] if last_iter else 0),
        "total_nets": int(nets_match.group(2)) if nets_match else (last_iter["total_nets"] if last_iter else 0),
        "converged": (converged_match.group(1) == "True") if converged_match else bool(last_iter),
        "iterations": last_iter["iter"] if last_iter else 0,
        "total_time_s": last_iter["total_time_s"] if last_iter else 0.0,
    }


def _compare_against_golden(log: logging.Logger, metrics: dict, golden_metrics: dict) -> None:
    gpu = golden_metrics.get("gpu", {})
    checks = [
        ("nets_routed", metrics["nets_routed"], golden_metrics.get("nets_routed")),
        ("total_nets", metrics["total_nets"], golden_metrics.get("total_nets")),
        ("iterations", metrics["iterations"], gpu.get("iterations_max")),
        ("total_time_s", metrics["total_time_s"], gpu.get("total_time_s_max")),
    ]

    log.warning("-" * 80)
    log.warning("HEADLESS VS GOLDEN COMPARISON")
    for name, actual, limit in checks:
        if limit is None:
            log.warning("%s: actual=%s (no golden threshold)", name, actual)
            continue
        if name in ("nets_routed", "total_nets"):
            status = "PASS" if actual == limit else "FAIL"
            log.warning("%s: actual=%s expected=%s => %s", name, actual, limit, status)
        else:
            status = "PASS" if actual <= limit else "WARN"
            log.warning("%s: actual=%.1f threshold=%.1f => %s", name, float(actual), float(limit), status)
    log.warning("converged: %s", metrics["converged"])
    log.warning("-" * 80)


def _validate_log_before_timeout(log: logging.Logger, timeout_s: int, log_file: Path | None) -> tuple[bool, dict]:
    if log_file is None or not log_file.exists():
        log.error("No log file found to validate routing completion")
        return False, {}

    text = log_file.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_log_completion_metrics(text)
    has_completion = parsed.get("has_routing_complete", False)
    completed_in_time = parsed.get("total_time_s", float("inf")) <= timeout_s

    if not has_completion:
        log.error("Log check failed: ROUTING COMPLETE banner not found in %s", log_file)
        return False, parsed
    if not completed_in_time:
        log.error(
            "Log check failed: routing completed at %.1fs, exceeding timeout %ds",
            parsed.get("total_time_s", 0.0),
            timeout_s,
        )
        return False, parsed

    log.warning(
        "Log check passed: completion found in %.1fs (timeout %ds)",
        parsed.get("total_time_s", 0.0),
        timeout_s,
    )
    return True, parsed


def run_headless(log_only: bool = False, explicit_log: Path | None = None) -> int:
    init_logging()
    log = logging.getLogger(__name__)

    golden_metrics = _load_golden_metrics()
    timeout_s = _compute_timeout_seconds(golden_metrics)
    log.warning("=" * 80)
    log.warning("HEADLESS ROUTING REGRESSION - TestBackplane.kicad_pcb")
    log.warning("Timeout: %ds (golden total_time_s_max +10%%)", timeout_s)
    log.warning("=" * 80)

    if log_only:
        log_file = explicit_log or _find_latest_log_file()
        ok, parsed = _validate_log_before_timeout(log, timeout_s, log_file)
        if ok:
            _compare_against_golden(log, parsed, golden_metrics)
            return 0
        return 1

    board_file = REPO_ROOT / "TestBoards" / "TestBackplane.kicad_pcb"
    if not board_file.exists():
        log.error("Board file not found: %s", board_file)
        return 1

    log.warning("Loading board from: %s", board_file)
    parser = KiCadFileParser()
    board = parser.load_board(str(board_file))
    if board is None:
        log.error("Failed to load board")
        return 1

    total_pads = sum(len(getattr(c, "pads", [])) for c in getattr(board, "components", []))
    log.warning("Board loaded: %d components, %d pads, %d nets", len(board.components), total_pads, len(board.nets))
    if total_pads < 10:
        log.error("Insufficient pads loaded: %d", total_pads)
        return 1

    cfg = PathFinderConfig()
    cfg.max_iterations = 50000
    router = UnifiedPathFinder(config=cfg, use_gpu=True)

    log.warning("Building routing graph...")
    router.initialize_graph(board)
    log.warning("Mapping pads to lattice...")
    router.map_all_pads(board)
    log.warning("Computing pad escape portals...")
    router.precompute_all_pad_escapes(board)

    log.warning("Starting routing for %d nets...", len(board.nets))
    start = time.monotonic()
    result = router.route_multiple_nets(board.nets)
    elapsed = time.monotonic() - start

    if not result:
        log.error("Routing failed - no result returned")
        return 1

    log.warning("=" * 80)
    log.warning("ROUTING COMPLETE")
    log.warning("Nets routed: %d/%d", result.get("nets_routed", 0), result.get("total_nets", 0))
    log.warning("Converged: %s", result.get("converged", False))
    log.warning("Total time: %.1fs", result.get("total_time_s", 0.0))
    log.warning("Iterations: %d", result.get("iterations", 0))
    log.warning("Elapsed wall time: %.1fs", elapsed)
    log.warning("=" * 80)

    log_file = explicit_log or _find_latest_log_file()
    ok, parsed = _validate_log_before_timeout(log, timeout_s, log_file)
    if not ok:
        return 1

    _compare_against_golden(log, parsed, golden_metrics)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless TestBackplane routing regression")
    parser.add_argument(
        "--log-only",
        action="store_true",
        help="Do not run routing; only validate latest (or provided) log against timeout/completion checks",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Specific log file to validate (used with --log-only or post-run validation)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run_headless(log_only=args.log_only, explicit_log=args.log_file))
