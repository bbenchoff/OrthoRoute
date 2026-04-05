"""
Regression tests for TestBackplane.kicad_pcb.

Test groups:
  A  — Log health + timing metrics (no routing required)
  A2 — Board-load verification against golden_board.json (always runs; parses .kicad_pcb directly)
  B  — Routing quality (requires routing_result fixture → full pipeline run or routing log)
  C  — Write-back verification (requires KiCad API, skipped without it)

Pass/fail policy:
  HARD FAIL  — test raises AssertionError; marks build broken
  SOFT WARN  — test calls pytest.warns or logs but never fails; metric tracking only

Only two tests are HARD FAIL:
  test_all_nets_routed
  test_convergence
All others are SOFT WARN (implemented via `warnings.warn`).
"""
import re
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
TEST_BOARD_FILE = REPO_ROOT / "TestBoards" / "TestBackplane.kicad_pcb"


@pytest.fixture(scope="module")
def board_file_text():
    """Raw .kicad_pcb text — always available, no routing needed."""
    if not TEST_BOARD_FILE.exists():
        pytest.skip(f"Test board not found: {TEST_BOARD_FILE}")
    return TEST_BOARD_FILE.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft(condition: bool, message: str) -> None:
    """Emit a UserWarning instead of failing when condition is False."""
    if not condition:
        warnings.warn(message, UserWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# Group A: Log health + timing metrics
# ---------------------------------------------------------------------------

class TestLogHealth:
    """Verify the log file contains no ERROR or CRITICAL entries."""

    def test_no_errors(self, log_content):
        """SOFT WARN: Unexpected ERROR lines in log."""
        errors = [ln for ln in log_content.splitlines()
                  if re.search(r"\bERROR\b", ln) and "[LOG]" not in ln]
        _soft(not errors,
              f"Log contains {len(errors)} ERROR line(s):\n" + "\n".join(errors[:5]))

    def test_no_criticals(self, log_content):
        """SOFT WARN: Any CRITICAL in the log is a serious regression."""
        crits = [ln for ln in log_content.splitlines() if re.search(r"\bCRITICAL\b", ln)]
        _soft(not crits,
              f"Log contains {len(crits)} CRITICAL line(s):\n" + "\n".join(crits[:5]))


class TestGPUMode:
    """Report and verify which compute mode the run used."""

    def test_gpu_mode_detected(self, log_content, gpu_mode):
        """SOFT WARN: Log the active compute mode. GPU preferred."""
        mode = "GPU" if gpu_mode else "CPU"
        _soft(gpu_mode, f"Routing ran in {mode} mode — GPU preferred for performance")

    def test_gpu_mode_matches_available_hardware(self, log_content, gpu_mode):
        """SOFT WARN: GPU=YES in log but cuda_dijkstra unavailable would be a config error."""
        if not gpu_mode:
            pytest.skip("CPU mode — skip GPU consistency check")
        cuda_ok = "CUDA-COMPILE" in log_content or "Compiled" in log_content
        _soft(cuda_ok, "GPU=YES but no CUDA kernel compilation found — possible config mismatch")


class TestIterationMetrics:
    """Parse [ITER N] log lines as an alternative source of timing data."""

    @pytest.fixture(scope="class")
    def iter_lines(self, log_content):
        pattern = re.compile(
            r"\[ITER\s+(\d+)\].*?nets=(\d+)/(\d+).*?iter=([0-9.]+)s.*?total=([0-9.]+)s"
        )
        rows = []
        for ln in log_content.splitlines():
            m = pattern.search(ln)
            if m:
                rows.append({
                    "iter": int(m.group(1)),
                    "nets_routed": int(m.group(2)),
                    "total_nets": int(m.group(3)),
                    "iter_time_s": float(m.group(4)),
                    "total_time_s": float(m.group(5)),
                })
        return rows

    def test_iter_lines_present(self, iter_lines):
        """SOFT WARN: No [ITER N] lines found – log may be wrong verbosity."""
        _soft(len(iter_lines) > 0, "No [ITER N] lines found in log")

    def test_iter_avg_time(self, iter_lines, active_metrics):
        """SOFT WARN: Average iteration time exceeds baseline threshold for active compute mode."""
        if not iter_lines:
            pytest.skip("No iteration data in log")
        avg = sum(r["iter_time_s"] for r in iter_lines) / len(iter_lines)
        limit = active_metrics["iter_avg_time_s_max"]
        _soft(avg <= limit, f"Avg iter time {avg:.1f}s > threshold {limit}s (performance regression?)")

    def test_iter_trend_not_exploding(self, iter_lines):
        """SOFT WARN: Iteration time grows >3× from first to last (runaway cost)."""
        if len(iter_lines) < 4:
            pytest.skip("Not enough iterations for trend check")
        first = iter_lines[0]["iter_time_s"]
        last = iter_lines[-1]["iter_time_s"]
        _soft(last <= first * 3, f"Iter time grew {last/first:.1f}× (first={first:.1f}s, last={last:.1f}s)")

    def test_total_time(self, iter_lines, active_metrics):
        """SOFT WARN: Total routing time exceeds baseline for active compute mode."""
        if not iter_lines:
            pytest.skip("No iteration data in log")
        total = iter_lines[-1]["total_time_s"]
        limit = active_metrics["total_time_s_max"]
        _soft(total <= limit, f"Total time {total:.0f}s > threshold {limit:.0f}s")


# ---------------------------------------------------------------------------
# Group A2: Board-load verification — parses .kicad_pcb directly
# ---------------------------------------------------------------------------

class TestBoardLoad:
    """
    Verify the test board file matches golden_board.json.

    Parses TestBackplane.kicad_pcb directly — never depends on log content.
    These tests always run regardless of log state.
    """

    def test_pad_count(self, board_file_text, golden_board):
        """HARD FAIL: Pad count in .kicad_pcb must match golden."""
        count = board_file_text.count("(pad ")
        assert count == golden_board["pads"], \
            f"Pad count {count} != golden {golden_board['pads']} (.kicad_pcb changed?)"

    def test_copper_layers(self, board_file_text, golden_board):
        """HARD FAIL: Copper layer count must match golden."""
        copper = set(re.findall(r'"((?:F|B)\.Cu|In\d+\.Cu)"', board_file_text))
        assert len(copper) == golden_board["copper_layers"], \
            f"Copper layers {len(copper)} != golden {golden_board['copper_layers']}"

    def test_existing_tracks(self, board_file_text, golden_board):
        """SOFT WARN: Segment count in .kicad_pcb should be within 5% of golden."""
        # KiCad 9 uses tab-indented "(segment\n\t\t(start..." — match without trailing space
        count = len(re.findall(r'\(segment\b', board_file_text))
        tol = int(golden_board["tracks_existing"] * golden_board["tolerance_tracks_pct"])
        _soft(abs(count - golden_board["tracks_existing"]) <= tol,
              f"Track segments {count} differs >5% from golden {golden_board['tracks_existing']}")

    def test_existing_vias(self, board_file_text, golden_board):
        """SOFT WARN: Via count in .kicad_pcb should be within 5% of golden."""
        count = len(re.findall(r'\(via ', board_file_text))
        tol = int(golden_board["vias_existing"] * golden_board["tolerance_vias_pct"])
        _soft(abs(count - golden_board["vias_existing"]) <= tol,
              f"Via count {count} differs >5% from golden {golden_board['vias_existing']}")

    def test_net_count(self, board_file_text, golden_board):
        """SOFT WARN: Total net declarations should match golden."""
        # Count (net N "name") declarations
        count = len(re.findall(r'\(net \d+', board_file_text))
        _soft(count >= golden_board.get("total_nets", 900),
              f"Net count {count} < golden {golden_board.get('total_nets', 900)}")

    def test_ipc_adapter_in_log(self, log_content):
        """SOFT WARN: Confirm preferred IPC adapter was selected (not SWIG/file fallback)."""
        used_ipc = "IPC" in log_content or "ipc" in log_content
        _soft(used_ipc, "IPC adapter may not have been used (check log for SWIG/file fallback)")


# ---------------------------------------------------------------------------
# Group B: Routing quality
# ---------------------------------------------------------------------------

class TestRoutingQuality:
    """
    Core routing quality checks.  HARD FAIL = 100% nets must route.
    All performance checks are SOFT WARNs.

    Uses `routing_result` (headless) when available, otherwise falls back to
    `log_routing_result` (parsed from the ORTHO_DEBUG=1 log).
    """

    REQUIRED_KEYS = [
        "success", "converged", "nets_routed", "total_nets",
        "iterations", "total_time_s", "iteration_metrics",
        "failed_nets", "overuse_sum", "overuse_edges",
        "barrel_conflicts", "excluded_nets", "excluded_net_ids",
        "error_code", "message",
    ]

    @pytest.fixture(scope="class")
    def result(self, routing_result, log_routing_result):
        """Best available routing result: headless run > log parse > skip."""
        r = routing_result if routing_result is not None else log_routing_result
        if r is None:
            pytest.skip("No routing result available — run OrthoRoute with ORTHO_DEBUG=1 first")
        return r

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_result_has_required_key(self, result, key):
        """HARD FAIL: routing result must contain all expected keys."""
        assert key in result, f"Missing key '{key}' in routing result"

    def test_all_nets_routed(self, result, golden_metrics):
        """HARD FAIL: Every routable net must be routed."""
        nets_routed = result.get("nets_routed", 0)
        total_nets = result.get("total_nets", golden_metrics["total_nets"])
        assert nets_routed == total_nets, \
            f"Only {nets_routed}/{total_nets} nets routed — routing regression!"

    def test_convergence(self, result):
        """HARD FAIL: Router must converge (overuse_final == 0)."""
        assert result.get("converged", False), \
            "Router did not converge (overuse edges remain)"

    def test_iteration_budget(self, result, active_metrics):
        """SOFT WARN: Used more iterations than baseline for active compute mode."""
        iters = result.get("iterations", 0)
        limit = active_metrics["iterations_max"]
        _soft(iters <= limit,
              f"Used {iters} iterations > baseline {limit} (algorithm efficiency regression?)")

    def test_total_time(self, result, active_metrics):
        """SOFT WARN: Total routing time exceeds baseline for active compute mode."""
        t = result.get("total_time_s", 0)
        limit = active_metrics["total_time_s_max"]
        _soft(t <= limit, f"Total time {t:.0f}s > baseline {limit:.0f}s")

    def test_overuse_final(self, result, active_metrics):
        """SOFT WARN: Final overuse count should be zero for a converged run."""
        overuse = result.get("overuse_final", result.get("overuse_sum", 0))
        _soft(overuse == 0, f"overuse_final={overuse} (should be 0 after convergence)")

    def test_iter_stability(self, result, active_metrics):
        """SOFT WARN: No single iteration should take >3× the active mode avg."""
        metrics = result.get("iteration_metrics", [])
        if not metrics:
            pytest.skip("No iteration_metrics in routing result")
        avg_limit = active_metrics["iter_avg_time_s_max"]
        spikes = [m for m in metrics if m["iter_time_s"] > avg_limit * 3]
        _soft(not spikes,
              f"{len(spikes)} iteration(s) took >3× avg limit ({avg_limit*3:.1f}s): "
              + ", ".join(f"iter {m['iter']}={m['iter_time_s']:.1f}s" for m in spikes[:3]))

    def test_no_barrel_conflicts(self, result):
        """SOFT WARN: Via barrel conflicts indicate geometry issues."""
        bc = result.get("barrel_conflicts", 0)
        _soft(bc == 0, f"{bc} barrel conflict(s) detected")

    def test_no_excluded_nets(self, result):
        """SOFT WARN: Excluded nets should stay at zero."""
        excl = result.get("excluded_nets", 0)
        _soft(excl == 0, f"{excl} net(s) excluded from routing")


# ---------------------------------------------------------------------------
# Group C: Write-back verification
# ---------------------------------------------------------------------------

@pytest.mark.requires_kicad
class TestWriteBack:
    """
    Verify that tracks and vias are actually written back to the board.
    Falls back to log-parsed writeback counts when headless routing unavailable.
    """

    @pytest.fixture(scope="class")
    def result(self, routing_result, log_routing_result):
        r = routing_result if routing_result is not None else log_routing_result
        if r is None:
            pytest.skip("No routing result available — run OrthoRoute with ORTHO_DEBUG=1 first")
        return r

    @pytest.fixture(scope="class")
    def pre_counts(self, golden_board):
        return {
            "tracks": golden_board["tracks_existing"],
            "vias": golden_board["vias_existing"],
        }

    def test_writeback_tracks_increased(self, result, pre_counts):
        """SOFT WARN: Track count must increase after routing."""
        post = result.get("tracks_written")
        if post is None:
            # Fall back to checking board_object if available
            pytest.skip("tracks_written not in result (headless routing not available)")
        _soft(post > pre_counts["tracks"],
              f"Track count did not increase: pre={pre_counts['tracks']}, post={post}")

    def test_writeback_vias_increased(self, result, pre_counts):
        """SOFT WARN: Via count must increase after routing."""
        post = result.get("vias_written")
        if post is None:
            pytest.skip("vias_written not in result (headless routing not available)")
        _soft(post > pre_counts["vias"],
              f"Via count did not increase: pre={pre_counts['vias']}, post={post}")
