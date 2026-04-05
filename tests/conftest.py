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

    Parses three layers of evidence:
      1. [ITER N] lines          — per-iteration timing + overuse
      2. End-of-run summary      — ROUTING COMPLETE / [CLEAN] / [FINAL] lines
      3. Write-back confirmation — "Applied N tracks and N vias to KiCad"

    Returns a routing_result-compatible dict, or None if the log has no
    routing completion marker.

    Recognised end-of-run patterns:
      WARNING - [CLEAN] All nets routed with zero overuse
      WARNING - ROUTING COMPLETE: All N nets routed successfully with zero overuse!
      WARNING - [FINAL] Edge routing converged (N barrel conflicts remain - acceptable)
      INFO    - Applied N tracks and N vias to KiCad          (✅ write-back)
    """
    if "ROUTING COMPLETE" not in text and "CONVERGED" not in text:
        return None

    # --- convergence + final iter summary ---
    # [ITER  67] nets=512/512  ✓ CONVERGED  edges=0  via_overuse=0%  barrel=379  iter=4.0s  total=761.2s
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

    # --- end-of-run summary block ---
    # [CLEAN] All nets routed with zero overuse
    clean_zero_overuse = bool(re.search(r"\[CLEAN\] All nets routed with zero overuse", text))

    # ROUTING COMPLETE: All N nets routed successfully with zero overuse!
    rc_m = re.search(r"ROUTING COMPLETE: All (\d+) nets routed successfully", text)
    routing_complete_nets = int(rc_m.group(1)) if rc_m else None

    # [FINAL] Edge routing converged (N barrel conflicts remain - acceptable)
    final_m = re.search(r"\[FINAL\] Edge routing converged \((\d+) barrel conflicts remain", text)
    final_barrel_conflicts = int(final_m.group(1)) if final_m else last["barrel"]

    # Strong convergence signal: both [CLEAN] and ROUTING COMPLETE present
    hard_converged = clean_zero_overuse or (rc_m is not None)

    # --- writeback confirmation ---
    # "Applied 4287 tracks and 2751 vias to KiCad"  (INFO from main_window)
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
        "excluded_nets": 0,
        "excluded_net_ids": [],
        "error_code": 0,
        "message": message,
        # writeback confirmation
        "tracks_written": tracks_out,
        "vias_written": vias_out,
        # end-of-run flags
        "clean_zero_overuse": clean_zero_overuse,
        "routing_complete_banner": rc_m is not None,
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
