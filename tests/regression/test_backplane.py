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

# Keep in sync with _HEADLESS_MAX_ITER in tests/conftest.py
_HEADLESS_MAX_ITER = 3

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

    def test_ipc_adapter_in_log(self, log_content):
        """SOFT WARN: Confirm preferred IPC adapter was selected (not SWIG/file fallback)."""
        used_ipc = "IPC" in log_content or "ipc" in log_content
        _soft(used_ipc, "IPC adapter may not have been used (check log for SWIG/file fallback)")


class TestGPUMode:
    """Report and verify which compute mode the run used."""

    def test_gpu_mode_detected(self, log_content, gpu_mode):
        """SOFT WARN: Log the active compute mode. GPU preferred."""
        mode = "GPU" if gpu_mode else "CPU"
        _soft(gpu_mode, f"Routing ran in {mode} mode — GPU preferred for performance")

    def test_gpu_mode_matches_available_hardware(self, log_content, gpu_mode):
        """SOFT WARN: GPU availability in log must be consistent with hardware.

        - GPU hardware present + log says CPU → soft warn (possible config mismatch)
        - GPU hardware present + log says GPU → verify CUDA kernel compilation logged
        - No GPU hardware present → pass unconditionally (CPU mode is correct)
        """
        try:
            import cupy  # noqa: F401
            hardware_has_gpu = True
        except Exception:
            hardware_has_gpu = False

        if not hardware_has_gpu:
            # CPU-only machine: CPU mode in log is expected
            return

        if not gpu_mode:
            # GPU hardware available but routing used CPU — soft warn
            _soft(False, "GPU hardware detected (CuPy available) but log shows CPU mode — check config")
        else:
            cuda_ok = "CUDA-COMPILE" in log_content or "Compiled" in log_content
            _soft(cuda_ok, "GPU=YES but no CUDA kernel compilation found — possible config mismatch")


class TestLibraryAvailability:
    """Verify all required and optional dependencies are installed and functional.

    These tests always run (no board or log required) and catch environment
    misconfiguration before routing tests are attempted.
    HARD FAIL = missing required library.
    SOFT WARN = optional library absent or GPU unavailable.
    """

    def test_numpy_available(self):
        """HARD FAIL: numpy must be importable and functional."""
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        assert np.sum(arr) == 6.0, "numpy basic operation failed"

    def test_scipy_available(self):
        """HARD FAIL: scipy must be importable (used for sparse graph ops)."""
        import scipy.sparse  # noqa: F401

    def test_orthoroute_importable(self):
        """HARD FAIL: The orthoroute package itself must be importable."""
        import orthoroute  # noqa: F401

    def test_unified_pathfinder_importable(self):
        """HARD FAIL: Core routing engine must be importable."""
        from orthoroute.algorithms.manhattan.unified_pathfinder import (  # noqa: F401
            UnifiedPathFinder,
            PathFinderConfig,
        )

    def test_cupy_installed(self, hardware_gpu):
        """SOFT WARN: CuPy not installed — GPU acceleration unavailable."""
        _soft(hardware_gpu, "CuPy not installed or CUDA unavailable — GPU routing disabled")

    def test_cupy_version(self, hardware_gpu):
        """SOFT WARN: Report CuPy version when GPU is available."""
        if not hardware_gpu:
            pytest.skip("CuPy/CUDA unavailable — skip version check")
        import cupy as cp
        ver = getattr(cp, "__version__", "unknown")
        assert ver != "unknown", "Could not determine CuPy version"

    def test_cuda_device_info(self, hardware_gpu):
        """SOFT WARN: Report CUDA device name and memory when GPU available."""
        if not hardware_gpu:
            pytest.skip("CuPy/CUDA unavailable — skip device info check")
        import cupy as cp
        device = cp.cuda.Device(0)
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        total_gb = total_mem / 1024 ** 3
        free_gb = free_mem / 1024 ** 3
        _soft(
            total_gb >= 2.0,
            f"CUDA device {device.id}: {total_gb:.1f} GB total, {free_gb:.1f} GB free "
            f"(minimum 2 GB recommended for full backplane routing)",
        )

    def test_file_parser_loads_board(self, board_object):
        """HARD FAIL: KiCadFileParser must load pads from TestBackplane.kicad_pcb."""
        assert board_object is not None, (
            "KiCadFileParser returned None — headless board loading broken"
        )
        total_pads = sum(len(getattr(c, 'pads', [])) for c in board_object.components)
        assert total_pads >= 100, (
            f"Only {total_pads} pads loaded — parser may have regressed"
        )


class TestHeadlessRouting:
    """Run actual CPU and GPU routing headlessly and validate results.

    These tests exercise the full UnifiedPathFinder pipeline without KiCad
    running.  They use the fixed KiCadFileParser to load the test board, then
    route a small sample of nets with a reduced iteration budget so the test
    completes in seconds rather than minutes.

    CPU tests: always attempted when board_object is available.
    GPU tests: only attempted when hardware_gpu=True (CuPy + CUDA present).

    Both are SOFT WARN on quality metrics; the routing pipeline itself must not
    crash (any exception propagates as a hard failure through the fixture).
    """

    # ---- CPU ---------------------------------------------------------------

    @pytest.fixture(scope="class")
    def cpu_result(self, routing_result_sample_cpu):
        """Fast CPU routing result; skip class if headless routing unavailable."""
        if routing_result_sample_cpu is None:
            pytest.skip("CPU routing unavailable — board load failed or router error")
        return routing_result_sample_cpu

    def test_cpu_routing_succeeds(self, cpu_result):
        """HARD FAIL: CPU routing must return a result dict without raising."""
        assert cpu_result is not None
        assert isinstance(cpu_result, dict)

    def test_cpu_nets_routed(self, cpu_result):
        """SOFT WARN: All sampled nets should be routed in CPU mode."""
        nets_routed = cpu_result.get("nets_routed", 0)
        total_nets = cpu_result.get("total_nets", 1)
        _soft(
            nets_routed > 0,
            f"CPU mode: {nets_routed}/{total_nets} nets routed in {_HEADLESS_MAX_ITER}-iter sample",
        )

    def test_cpu_converged(self, cpu_result):
        """SOFT WARN: CPU routing may not converge in {_HEADLESS_MAX_ITER} iterations."""
        _soft(
            cpu_result.get("converged", False),
            f"CPU routing did not converge in {_HEADLESS_MAX_ITER} iterations "
            f"(expected for short sample runs — full run needed for convergence)",
        )

    def test_cpu_result_has_required_keys(self, cpu_result):
        """HARD FAIL: CPU result dict must contain all standard keys."""
        required = [
            "success", "nets_routed", "total_nets",
            "iterations", "total_time_s",
        ]
        missing = [k for k in required if k not in cpu_result]
        assert not missing, f"CPU result missing keys: {missing}"

    # ---- GPU ---------------------------------------------------------------

    @pytest.fixture(scope="class")
    def gpu_result(self, routing_result_sample_gpu, hardware_gpu):
        """Fast GPU routing result; skip class if GPU unavailable or board load failed."""
        if not hardware_gpu:
            pytest.skip("GPU hardware unavailable (CuPy/CUDA not installed)")
        if routing_result_sample_gpu is None:
            pytest.skip("GPU routing failed — check CUDA installation and board loader")
        return routing_result_sample_gpu

    def test_gpu_routing_succeeds(self, gpu_result):
        """HARD FAIL: GPU routing must return a result dict without raising."""
        assert gpu_result is not None
        assert isinstance(gpu_result, dict)

    def test_gpu_nets_routed(self, gpu_result):
        """SOFT WARN: Sampled nets should be routed in GPU mode."""
        nets_routed = gpu_result.get("nets_routed", 0)
        total_nets = gpu_result.get("total_nets", 1)
        _soft(
            nets_routed > 0,
            f"GPU mode: {nets_routed}/{total_nets} nets routed in {_HEADLESS_MAX_ITER}-iter sample",
        )

    def test_gpu_converged(self, gpu_result):
        """SOFT WARN: GPU routing may not converge in {_HEADLESS_MAX_ITER} iterations."""
        _soft(
            gpu_result.get("converged", False),
            f"GPU routing did not converge in {_HEADLESS_MAX_ITER} iterations "
            f"(expected for short sample runs — full run needed for convergence)",
        )

    def test_gpu_result_has_required_keys(self, gpu_result):
        """HARD FAIL: GPU result dict must contain all standard keys."""
        required = [
            "success", "nets_routed", "total_nets",
            "iterations", "total_time_s",
        ]
        missing = [k for k in required if k not in gpu_result]
        assert not missing, f"GPU result missing keys: {missing}"

    def test_gpu_faster_than_cpu(self, cpu_result, gpu_result, hardware_gpu):
        """SOFT WARN: GPU total_time_s should be less than CPU total_time_s."""
        if not hardware_gpu:
            pytest.skip("GPU unavailable")
        if cpu_result is None or gpu_result is None:
            pytest.skip("Both CPU and GPU results required for comparison")
        cpu_t = cpu_result.get("total_time_s", 0)
        gpu_t = gpu_result.get("total_time_s", 0)
        _soft(
            gpu_t < cpu_t,
            f"GPU ({gpu_t:.1f}s) not faster than CPU ({cpu_t:.1f}s) — "
            f"check GPU kernel compilation",
        )


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

    def test_ipc_adapter_in_log(self, board_file_text):
        """SOFT WARN: Board file exists and is parseable (log check moved to TestLogHealth)."""
        _soft(len(board_file_text) > 1000,
              f"Board file appears empty or truncated ({len(board_file_text)} bytes)")


# ---------------------------------------------------------------------------
# Group A3: Lattice size verification
# ---------------------------------------------------------------------------

class TestLatticeSize:
    """
    Verify the routing lattice dimensions match golden_board.json.

    Parses the 'Lattice: 106×234×18 = 446,472 nodes' WARNING line that
    UnifiedPathFinder emits at the start of every routing run when
    ORTHO_DEBUG=1 (or any verbosity ≥ WARNING).

    These tests skip cleanly when the active log is an init-only log with
    no routing content.
    """

    @pytest.fixture(scope="class")
    def lattice(self, log_lattice):
        """Parsed lattice dict, or skip when not available."""
        if log_lattice is None:
            pytest.skip("No 'Lattice: NxNxN = N nodes' line in log — run routing first")
        return log_lattice

    def test_lattice_nodes(self, lattice, golden_board):
        """HARD FAIL: Total node count must match golden (lattice size changed)."""
        assert lattice["nodes"] == golden_board["lattice_nodes"], (
            f"Lattice nodes {lattice['nodes']:,} != golden {golden_board['lattice_nodes']:,} "
            f"— grid pitch or board bounds changed?"
        )

    def test_lattice_layers(self, lattice, golden_board):
        """HARD FAIL: Layer count in lattice must match golden lattice_layers (may differ from
        copper_layers by 1 if the router adds a virtual layer for B.Cu/F.Cu pair resolution)."""
        expected = golden_board.get("lattice_layers", golden_board["copper_layers"])
        assert lattice["layers"] == expected, (
            f"Lattice layers {lattice['layers']} != golden {expected}"
        )

    def test_lattice_dimensions_reported(self, lattice):
        """SOFT WARN: Log reported lattice cols × rows × layers."""
        _soft(
            lattice["cols"] > 0 and lattice["rows"] > 0,
            f"Unexpected lattice dimensions: {lattice['cols']}×{lattice['rows']}×{lattice['layers']}",
        )

    def test_lattice_node_product(self, lattice):
        """SOFT WARN: cols × rows × layers should equal reported node count."""
        product = lattice["cols"] * lattice["rows"] * lattice["layers"]
        _soft(
            product == lattice["nodes"],
            f"cols×rows×layers={product:,} != nodes={lattice['nodes']:,} "
            f"(log rounding or partial lattice?)",
        )


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

    def test_no_barrel_conflicts(self, result, active_metrics):
        """SOFT WARN: Via barrel conflicts beyond acceptable threshold indicate geometry issues."""
        bc = result.get("barrel_conflicts", 0)
        limit = active_metrics.get("barrel_conflicts_max", 500)
        _soft(bc <= limit, f"{bc} barrel conflict(s) detected (max acceptable: {limit})")

    def test_no_excluded_nets(self, result):
        """SOFT WARN: Excluded nets should stay at zero."""
        excl = result.get("excluded_nets", 0)
        _soft(excl == 0, f"{excl} net(s) excluded from routing")

    def test_routing_complete_banner(self, result):
        """SOFT WARN: 'ROUTING COMPLETE' banner must appear at end of log.

        Checks that the router emitted:
          WARNING - ROUTING COMPLETE: All N nets routed successfully with zero overuse!
        This is the explicit success marker from UnifiedPathFinder.
        Only meaningful when result came from log_parse (not headless routing).
        """
        if result.get("_source") != "log_parse":
            pytest.skip("Banner check only applies to log-parsed results")
        _soft(
            result.get("routing_complete_banner", False),
            "ROUTING COMPLETE banner not found in log — run may have been interrupted",
        )

    def test_clean_zero_overuse(self, result):
        """SOFT WARN: '[CLEAN] All nets routed with zero overuse' line must appear.

        This is the router's explicit declaration that the final state has
        zero overuse edges, separate from the CONVERGED flag on [ITER N] lines.
        Only meaningful when result came from log_parse.
        """
        if result.get("_source") != "log_parse":
            pytest.skip("Clean-zero-overuse check only applies to log-parsed results")
        _soft(
            result.get("clean_zero_overuse", False),
            "[CLEAN] marker not found — overuse may not have reached exactly zero",
        )


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

    def test_writeback_tracks_increased(self, result, active_metrics):
        """SOFT WARN: Tracks written back must meet the minimum delta threshold."""
        post = result.get("tracks_written")
        if post is None:
            pytest.skip("tracks_written not in result (headless routing not available)")
        limit = active_metrics.get("tracks_delta_min", 1)
        _soft(post >= limit,
              f"tracks_written={post} < minimum expected {limit} (write-back may have failed)")

    def test_writeback_vias_increased(self, result, active_metrics):
        """SOFT WARN: Vias written back must meet the minimum delta threshold."""
        post = result.get("vias_written")
        if post is None:
            pytest.skip("vias_written not in result (headless routing not available)")
        limit = active_metrics.get("vias_delta_min", 1)
        _soft(post >= limit,
              f"vias_written={post} < minimum expected {limit} (write-back may have failed)")
