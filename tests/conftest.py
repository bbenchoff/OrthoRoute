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
    """Path to the most recent log file, or the latest.log symlink."""
    # Check every candidate directory for latest.log or timestamped logs
    for d in LOG_SEARCH_DIRS:
        latest = d / "latest.log"
        if latest.exists():
            return latest
    all_runs = []
    for d in LOG_SEARCH_DIRS:
        all_runs.extend(d.glob("run_*.log"))
    if not all_runs:
        pytest.skip("No log files found – run OrthoRoute first (checked: " +
                    ", ".join(str(d) for d in LOG_SEARCH_DIRS) + ")")
    return max(all_runs, key=lambda p: p.stat().st_mtime)


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

    Uses the same file-parser adapter the CLI uses, so no running KiCad
    process is required.  Skips if the parser returns an empty board (< 10
    pads), which happens when the S-expression parser cannot handle the board
    — in that case Group B/C routing tests will be skipped automatically.
    """
    from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser
    parser = KiCadFileParser()
    board = parser.load_board(str(board_file))
    if board is None:
        pytest.skip(f"KiCadFileParser could not load {board_file}")
    # Count pads across all nets (same source the router uses)
    total_pads = sum(len(getattr(n, 'pads', [])) for n in getattr(board, 'nets', []))
    if total_pads < 10:
        pytest.skip(
            f"KiCadFileParser loaded only {total_pads} pads from {board_file.name} — "
            "board parsing failed headlessly; run via KiCad to enable routing tests"
        )
    return board


@pytest.fixture(scope="session")
def router(board_object):
    """
    Create and initialise a UnifiedPathFinder instance for the test board.
    """
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        UnifiedPathFinder,
        PathFinderConfig,
    )
    _router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=False)
    _router.initialize_graph(board_object)
    _router.map_all_pads(board_object)
    return _router


@pytest.fixture(scope="session")
def routing_result(board_object, router):
    """
    Run routing once and cache the result for the entire test session.

    The `router` fixture already calls initialize_graph() and map_all_pads(),
    so we call route_multiple_nets() directly to avoid double-initialisation.
    All regression tests that need routing results should depend on this fixture.
    """
    return router.route_multiple_nets(board_object.nets)


# ---------------------------------------------------------------------------
# GPU / CPU mode detection
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
