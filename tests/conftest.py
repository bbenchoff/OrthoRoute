"""
Shared pytest fixtures for OrthoRoute regression and unit tests.

Session-scoped fixtures capture one full routing run so all tests share the
result without re-routing.  Fixture dependency order:
  board_file → board_object → router → routing_result
  log_path   → log_content
  golden_board / golden_metrics ← loaded from JSON files next to this module
"""
import json
import logging
import os
import re
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths (/tests/ lives at repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
TEST_BOARD = REPO_ROOT / "TestBoards" / "TestBackplane.kicad_pcb"
GOLDEN_BOARD_JSON = Path(__file__).parent / "regression" / "golden_board.json"
GOLDEN_METRICS_JSON = Path(__file__).parent / "regression" / "golden_metrics.json"
LOG_DIR = REPO_ROOT / "logs"

# Additional log search locations (KiCad plugin folder, env override)
_PLUGIN_LOG_DIR = Path(
    os.environ.get(
        "ORTHO_LOG_DIR",
        str(Path.home() / "OneDrive - Rockwell Automation, Inc"
            / "Simulation tools" / "KiCad" / "9.0" / "3rdparty" / "plugins"
            / "com_github_bbenchoff_orthoroute" / "logs"),
    )
)

LOG_SEARCH_DIRS = [d for d in [LOG_DIR, _PLUGIN_LOG_DIR] if d.exists()]


# ---------------------------------------------------------------------------
# Golden fixtures (loaded from JSON reference files)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_board():
    """Expected board signature for TestBackplane.kicad_pcb."""
    with open(GOLDEN_BOARD_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def golden_metrics():
    """Baseline routing metrics for TestBackplane.kicad_pcb."""
    with open(GOLDEN_METRICS_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Log fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def log_path():
    """
    Path to the best available log file.

    Preference order:
      1. Largest file (by byte size) across all search dirs — most content wins.
      2. Any run_*.log if latest.log is absent.

    Rationale: latest.log gets overwritten by every KiCad init (even without
    routing), so a tiny 143-byte init log would beat a 63,000-line routing log
    if we sorted by mtime. Size is a better proxy for 'most useful'.
    """
    all_logs = []
    for d in LOG_SEARCH_DIRS:
        all_logs.extend(d.glob("*.log"))
    if not all_logs:
        pytest.skip(
            "No log files found – run OrthoRoute first (checked: " +
            ", ".join(str(d) for d in LOG_SEARCH_DIRS) + ")"
        )
    return max(all_logs, key=lambda p: p.stat().st_size)


@pytest.fixture(scope="session")
def log_content(log_path):
    """Full text of the selected log file."""
    return log_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Board / router / routing result fixtures  (require KiCad Python API)
# ---------------------------------------------------------------------------

def _kicad_api_available() -> bool:
    """Return True when the KiCad IPC client is importable."""
    try:
        import kiapi  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pcbnew  # noqa: F401
        return True
    except ImportError:
        return False


requires_kicad = pytest.mark.skipif(
    not _kicad_api_available(),
    reason="KiCad Python API not available in this environment",
)


@pytest.fixture(scope="session")
def board_file():
    """Absolute path to TestBackplane.kicad_pcb."""
    if not TEST_BOARD.exists():
        pytest.skip(f"Test board not found: {TEST_BOARD}")
    return TEST_BOARD


@pytest.fixture(scope="session")
def board_object(board_file):
    """
    Load the test board via the KiCadFileParser.

    Returns None (rather than skipping) when the parser cannot load pads,
    so higher-level fixtures can fall back to log-based results instead of
    skipping the entire session.
    """
    try:
        from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
        parser = KiCadFileParser()
        board = parser.load_board(str(board_file))
    except Exception:
        return None
    if board is None:
        return None
    total_pads = sum(len(getattr(n, 'pads', [])) for n in getattr(board, 'nets', []))
    if total_pads < 10:
        return None
    return board


@pytest.fixture(scope="session")
def router(board_object):
    """
    Create and initialise a UnifiedPathFinder instance for the test board.
    Returns None when board_object is unavailable.
    """
    if board_object is None:
        return None
    try:
        from orthoroute.algorithms.manhattan.unified_pathfinder import (
            UnifiedPathFinder,
            PathFinderConfig,
        )
        _router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
        _router.initialize_graph(board_object)
        _router.map_all_pads(board_object)
        return _router
    except Exception:
        return None


@pytest.fixture(scope="session")
def routing_result(board_object, router):
    """
    Run routing once and cache the result for the entire test session.
    Returns None when headless routing is unavailable (board parse failed).
    """
    if board_object is None or router is None:
        return None
    try:
        return router.route_multiple_nets(board_object.nets)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Log-based routing result (fallback when headless routing is unavailable)
# ---------------------------------------------------------------------------

def _parse_routing_result_from_log(text: str) -> dict | None:
    """
    Extract routing summary metrics from a ORTHO_DEBUG=1 log.

    Returns a routing_result-compatible dict, or None if the log has no
    routing completion marker.
    """
    if "ROUTING COMPLETE" not in text and "CONVERGED" not in text:
        return None

    # --- convergence + final iter summary ---
    # [ITER  65] nets=512/512  ✓ CONVERGED  edges=0 ... barrel=444  iter=4.6s  total=717.5s
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

    # --- writeback tracks / vias ---
    # Applied 4290 tracks and 2754 vias to KiCad
    wb_m = re.search(r"Applied (\d+) tracks and (\d+) vias", text)
    tracks_out = int(wb_m.group(1)) if wb_m else None
    vias_out = int(wb_m.group(2)) if wb_m else None

    return {
        "success": True,
        "converged": last.get("converged", False),
        "nets_routed": last["nets_routed"],
        "total_nets": last["total_nets"],
        "iterations": last["iter"],
        "total_time_s": last["total_time_s"],
        "iteration_metrics": [
            {"iter": r["iter"], "iter_time_s": r["iter_time_s"]} for r in iter_rows
        ],
        "failed_nets": last["total_nets"] - last["nets_routed"],
        "overuse_sum": last["overuse_edges"],
        "overuse_edges": last["overuse_edges"],
        "barrel_conflicts": last["barrel"],
        "excluded_nets": 0,
        "excluded_net_ids": [],
        "error_code": 0,
        "message": f"Parsed from log: {last['nets_routed']}/{last['total_nets']} nets routed",
        # writeback bonus fields
        "tracks_written": tracks_out,
        "vias_written": vias_out,
        "_source": "log_parse",
    }


@pytest.fixture(scope="session")
def log_routing_result(log_content) -> dict | None:
    """Routing result dict parsed from the log, or None if log has no routing data."""
    return _parse_routing_result_from_log(log_content)


# ---------------------------------------------------------------------------
# GPU / CPU mode detection
# ---------------------------------------------------------------------------
# Lattice size parsed from log
# ---------------------------------------------------------------------------

def _parse_lattice_from_log(text: str) -> dict | None:
    """
    Extract lattice dimensions from a log line such as:
      WARNING - Lattice: 106×234×18 = 446,472 nodes

    Returns dict with keys: cols, rows, layers, nodes, or None if not found.
    """
    m = re.search(
        r"Lattice:\s*(\d+)[×x*](\d+)[×x*](\d+)\s*=\s*([\d,]+)\s*nodes",
        text,
    )
    if not m:
        return None
    return {
        "cols": int(m.group(1)),
        "rows": int(m.group(2)),
        "layers": int(m.group(3)),
        "nodes": int(m.group(4).replace(",", "")),
    }


@pytest.fixture(scope="session")
def log_lattice(log_content) -> dict | None:
    """
    Lattice dimensions parsed from the log.

    Returns dict with keys {cols, rows, layers, nodes}, or None if the log
    has no 'Lattice: NxNxN = N nodes' line (init-only log or non-debug run).
    """
    return _parse_lattice_from_log(log_content)


# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_mode(log_content) -> bool:
    """True if the log shows GPU=YES (PathFinder ran with GPU acceleration)."""
    return bool(re.search(r"PathFinder loaded \(GPU=YES\)", log_content))


@pytest.fixture(scope="session")
def active_metrics(golden_metrics, gpu_mode) -> dict:
    """
    Return the right performance threshold block from golden_metrics.json.

    - If GPU=YES in log  → use golden_metrics["gpu"]
    - If GPU=NO in log   → use golden_metrics["cpu"]
    - Top-level keys are always available as a fallback.
    """
    block_key = "gpu" if gpu_mode else "cpu"
    block = golden_metrics.get(block_key, {})
    # Merge: block values take priority over top-level fallbacks.
    merged = dict(golden_metrics)
    merged.update(block)
    return merged
