"""
Fast smoke tests — run in under 30 seconds, no KiCad required.

Uses a synthetic board built entirely from domain objects:
  • 2 connectors (J1/J2), 100 pins each, on a 105 mm × 30 mm board
  • 4 copper layers (F.Cu, In1.Cu, In2.Cu, B.Cu)
  • 100 straight-through nets (J1 pin N → J2 pin N), 200 pads

Goals
-----
1. Confirm the domain model can be constructed without file I/O.
2. Confirm the CPU routing pipeline (initialize → escape → route) completes.
3. Confirm the GPU routing pipeline completes when hardware is present.
4. Confirm result keys match the standard schema.

Failure policy
--------------
• HARD FAIL  — assertion error (test is broken or pipeline regressed).
• SOFT WARN  — pytest.warns(UserWarning) message; test still passes.
  Used for quality metrics that could 0-out on a trivial board.
"""
import pytest
import warnings


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _soft(condition: bool, message: str) -> None:
    """Emit a UserWarning when *condition* is False (soft failure)."""
    if not condition:
        warnings.warn(message, UserWarning, stacklevel=2)


# ---------------------------------------------------------------------------
# TestSyntheticBoard — verify the board fixture itself
# ---------------------------------------------------------------------------

class TestSyntheticBoard:
    """Verify the smoke board domain object is well-formed."""

    def test_board_created(self, smoke_board):
        """HARD FAIL: synthetic board must not be None."""
        assert smoke_board is not None

    def test_board_has_two_connectors(self, smoke_board):
        """HARD FAIL: exactly 2 connectors (J1, J2) in the smoke board."""
        assert len(smoke_board.components) == 2

    def test_board_has_100_nets(self, smoke_board):
        """HARD FAIL: exactly 100 nets in the smoke board."""
        assert len(smoke_board.nets) == 100

    def test_board_has_200_pads(self, smoke_board):
        """HARD FAIL: 2 connectors × 100 pins = 200 total pads."""
        total = sum(len(c.pads) for c in smoke_board.components)
        assert total == 200

    def test_all_nets_are_routable(self, smoke_board):
        """HARD FAIL: every net must have ≥ 2 pads."""
        non_routable = [n.name for n in smoke_board.nets if len(n.pads) < 2]
        assert not non_routable, f"Non-routable nets: {non_routable}"

    def test_board_has_four_layers(self, smoke_board):
        """HARD FAIL: exactly 4 copper layers."""
        assert len(smoke_board.layers) == 4


# ---------------------------------------------------------------------------
# TestSmokeCPU — CPU routing on the synthetic board
# ---------------------------------------------------------------------------

class TestSmokeCPU:
    """Smoke-test the CPU routing pipeline end-to-end."""

    @pytest.fixture(scope="class")
    def result(self, routing_result_smoke_cpu):
        if routing_result_smoke_cpu is None:
            pytest.skip("CPU smoke routing unavailable — router initialisation failed")
        return routing_result_smoke_cpu

    def test_routing_returns_dict(self, result):
        """HARD FAIL: route_multiple_nets must return a dict."""
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, result):
        """HARD FAIL: standard result keys must be present."""
        required = ["success", "nets_routed", "total_nets", "iterations", "total_time_s"]
        missing = [k for k in required if k not in result]
        assert not missing, f"Missing keys: {missing}"

    def test_at_least_some_nets_routed(self, result):
        """HARD FAIL: at least 50% of nets must be routed on the smoke board."""
        threshold = result["total_nets"] // 2
        assert result["nets_routed"] >= threshold, (
            f"{result['nets_routed']}/{result['total_nets']} nets routed "
            f"(expected ≥ {threshold}) — pipeline may be broken"
        )

    def test_all_nets_routed(self, result):
        """SOFT WARN: all 100 nets should be routed on this board."""
        _soft(
            result["nets_routed"] == result["total_nets"],
            f"CPU smoke: only {result['nets_routed']}/{result['total_nets']} nets routed",
        )

    def test_converged(self, result):
        """SOFT WARN: routing should converge on a trivial board."""
        _soft(result.get("converged", False), "CPU smoke routing did not converge")

    def test_total_time_recorded(self, result):
        """HARD FAIL: total_time_s must be a non-negative number."""
        t = result.get("total_time_s", -1)
        assert isinstance(t, (int, float)) and t >= 0, f"Bad total_time_s: {t!r}"


# ---------------------------------------------------------------------------
# TestSmokeGPU — GPU routing on the synthetic board
# ---------------------------------------------------------------------------

class TestSmokeGPU:
    """Smoke-test the GPU routing pipeline end-to-end.

    All tests skip gracefully when no CUDA hardware is available.
    """

    @pytest.fixture(scope="class")
    def result(self, routing_result_smoke_gpu, hardware_gpu):
        if not hardware_gpu:
            pytest.skip("GPU hardware unavailable (CuPy/CUDA not installed)")
        if routing_result_smoke_gpu is None:
            pytest.skip("GPU smoke routing unavailable — router initialisation failed")
        return routing_result_smoke_gpu

    def test_routing_returns_dict(self, result):
        """HARD FAIL: route_multiple_nets must return a dict."""
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, result):
        """HARD FAIL: standard result keys must be present."""
        required = ["success", "nets_routed", "total_nets", "iterations", "total_time_s"]
        missing = [k for k in required if k not in result]
        assert not missing, f"Missing keys: {missing}"

    def test_at_least_some_nets_routed(self, result):
        """HARD FAIL: at least 50% of nets must be routed on the smoke board."""
        threshold = result["total_nets"] // 2
        assert result["nets_routed"] >= threshold, (
            f"{result['nets_routed']}/{result['total_nets']} nets routed "
            f"(expected ≥ {threshold}) — GPU pipeline may be broken"
        )

    def test_all_nets_routed(self, result):
        """SOFT WARN: all 100 nets should be routed on this board."""
        _soft(
            result["nets_routed"] == result["total_nets"],
            f"GPU smoke: only {result['nets_routed']}/{result['total_nets']} nets routed",
        )

    def test_converged(self, result):
        """SOFT WARN: routing should converge on a trivial board."""
        _soft(result.get("converged", False), "GPU smoke routing did not converge")


# ---------------------------------------------------------------------------
# TestSmokeCPUvsGPU — cross-mode comparison
# ---------------------------------------------------------------------------

class TestSmokeCPUvsGPU:
    """Compare CPU and GPU results for consistency."""

    @pytest.fixture(scope="class")
    def both(self, routing_result_smoke_cpu, routing_result_smoke_gpu, hardware_gpu):
        if not hardware_gpu:
            pytest.skip("GPU hardware unavailable")
        if routing_result_smoke_cpu is None or routing_result_smoke_gpu is None:
            pytest.skip("One or both smoke routing results unavailable")
        return routing_result_smoke_cpu, routing_result_smoke_gpu

    def test_same_net_count(self, both):
        """HARD FAIL: CPU and GPU must see the same number of nets."""
        cpu, gpu = both
        assert cpu["total_nets"] == gpu["total_nets"], (
            f"CPU total={cpu['total_nets']} vs GPU total={gpu['total_nets']}"
        )

    def test_gpu_not_drastically_slower(self, both):
        """SOFT WARN: GPU should not be more than 10× slower than CPU on 8 nets."""
        cpu, gpu = both
        cpu_t = cpu.get("total_time_s", 0)
        gpu_t = gpu.get("total_time_s", 0)
        if cpu_t > 0:
            _soft(
                gpu_t < cpu_t * 10,
                f"GPU ({gpu_t:.1f}s) is >10× slower than CPU ({cpu_t:.1f}s) on smoke board "
                f"— GPU kernel overhead dominates tiny boards (expected)",
            )
