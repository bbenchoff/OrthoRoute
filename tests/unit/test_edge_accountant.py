"""Unit tests for EdgeAccountant - Cost management and congestion tracking.

Tests EMA smoothing, history penalty accumulation, decay, and layer bias
to ensure correct PathFinder cost computation and convergence behavior.
"""
import pytest
import numpy as np

# Try importing cupy for GPU tests
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_accountant():
    """EdgeAccountant with 10 edges for basic testing (CPU only)."""
    from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
    return EdgeAccountant(num_edges=10, use_gpu=False)


@pytest.fixture
def accountant_with_usage():
    """EdgeAccountant with pre-populated usage for cost testing."""
    from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
    acc = EdgeAccountant(num_edges=10, use_gpu=False)
    
    # Set up usage: edges 0-2 overused, 3-5 at capacity, 6-9 under capacity
    acc.present[:3] = 2.0  # overused (capacity=1)
    acc.present[3:6] = 1.0  # at capacity
    acc.present[6:] = 0.5   # under capacity
    
    # Initialize present_ema to match present (steady state)
    acc.present_ema = acc.present.copy()
    
    return acc


@pytest.fixture
def base_costs():
    """Uniform base costs for testing."""
    return np.array([1.0] * 10, dtype=np.float32)


@pytest.fixture(params=[False, True] if GPU_AVAILABLE else [False])
def accountant_cpu_and_gpu(request):
    """Parametrized fixture for both CPU and GPU testing."""
    from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
    use_gpu = request.param
    if use_gpu and not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    return EdgeAccountant(num_edges=10, use_gpu=use_gpu)


# ============================================================================
# TestEdgeAccountantInit - Initialization and basic operations
# ============================================================================

class TestEdgeAccountantInit:
    """Test EdgeAccountant construction and basic operations."""

    def test_accountant_constructs_with_correct_size(self, small_accountant):
        """Test that EdgeAccountant initializes arrays with correct size."""
        assert small_accountant.E == 10
        assert len(small_accountant.present) == 10
        assert len(small_accountant.present_ema) == 10
        assert len(small_accountant.history) == 10
        assert len(small_accountant.capacity) == 10

    def test_accountant_arrays_initialized_to_zero(self, small_accountant):
        """Test that usage arrays start at zero."""
        assert np.allclose(small_accountant.present, 0.0)
        assert np.allclose(small_accountant.present_ema, 0.0)
        assert np.allclose(small_accountant.history, 0.0)

    def test_accountant_capacity_defaults_to_one(self, small_accountant):
        """Test that edge capacity defaults to 1."""
        assert np.allclose(small_accountant.capacity, 1.0)

    def test_commit_path_increments_usage(self, small_accountant):
        """Test that commit_path() increments edge usage correctly."""
        path_edges = [0, 1, 2]
        small_accountant.commit_path(path_edges)
        
        assert small_accountant.present[0] == 1.0
        assert small_accountant.present[1] == 1.0
        assert small_accountant.present[2] == 1.0
        assert np.allclose(small_accountant.present[3:], 0.0)

    def test_commit_path_handles_repeated_edges(self, small_accountant):
        """Test that commit_path() accumulates usage for reused edges."""
        path_edges = [0, 1, 0, 2]  # edge 0 used twice
        small_accountant.commit_path(path_edges)
        
        assert small_accountant.present[0] == 2.0
        assert small_accountant.present[1] == 1.0
        assert small_accountant.present[2] == 1.0


# ============================================================================
# TestPresentEMA - Exponential moving average smoothing
# ============================================================================

class TestPresentEMA:
    """Test update_present_ema() exponential moving average calculation."""

    def test_present_ema_first_update_converges_toward_present(self, small_accountant):
        """Test that first EMA update moves toward current present value.
        
        Formula: present_ema = beta * present + (1 - beta) * present_ema_old
        With beta=0.6 and present_ema_old=0, should reach 60% of present.
        """
        small_accountant.present[0] = 10.0
        small_accountant.present_ema[0] = 0.0
        
        small_accountant.update_present_ema(beta=0.6)
        
        expected = 0.6 * 10.0 + 0.4 * 0.0  # 6.0
        assert np.isclose(small_accountant.present_ema[0], expected)

    def test_present_ema_smooths_oscillations(self, small_accountant):
        """Test that EMA smooths bang-bang oscillations.
        
        Simulates usage alternating 0→10→0→10, EMA should stay between.
        """
        beta = 0.6
        
        # Iteration 1: present=10
        small_accountant.present[0] = 10.0
        small_accountant.update_present_ema(beta=beta)
        ema_1 = small_accountant.present_ema[0]
        assert 5.0 < ema_1 < 10.0, f"EMA should be between 5-10, got {ema_1}"
        
        # Iteration 2: present=0 (route removed)
        small_accountant.present[0] = 0.0
        small_accountant.update_present_ema(beta=beta)
        ema_2 = small_accountant.present_ema[0]
        assert ema_2 < ema_1, f"EMA should decrease when present drops"
        assert ema_2 > 0.0, f"EMA should stay positive (smoothing)"

    def test_present_ema_beta_effect(self, small_accountant):
        """Test that higher beta means faster response (less smoothing).
        
        Formula: present_ema = beta*present + (1-beta)*present_ema_old
        beta=0.9 (low smoothing) → fast response (90% of new value)
        beta=0.1 (high smoothing) → slow response (10% of new value)
        """
        small_accountant.present[0] = 10.0
        small_accountant.present_ema[0] = 0.0
        
        # High beta (fast response, low smoothing)
        small_accountant.update_present_ema(beta=0.9)
        ema_high = small_accountant.present_ema[0]
        
        # Reset for low beta (slow response, high smoothing)
        small_accountant.present_ema[0] = 0.0
        small_accountant.update_present_ema(beta=0.1)
        ema_low = small_accountant.present_ema[0]
        
        # High beta should respond faster (reach closer to present=10)
        assert ema_high > ema_low, f"Higher beta should respond faster: {ema_high} vs {ema_low}"

    def test_present_ema_steady_state(self, small_accountant):
        """Test that EMA converges to present value in steady state."""
        small_accountant.present[0] = 5.0
        small_accountant.present_ema[0] = 5.0
        
        small_accountant.update_present_ema(beta=0.6)
        
        # With present=5, ema=5: 0.6*5 + 0.4*5 = 5
        assert np.isclose(small_accountant.present_ema[0], 5.0)


# ============================================================================
# TestHistoryUpdate - History penalty accumulation and decay
# ============================================================================

class TestHistoryUpdate:
    """Test update_history() accumulation, capping, and decay."""

    def test_history_accumulates_for_overused_edges(self, accountant_with_usage):
        """Test that history penalty accumulates on overused edges.
        
        Edges with usage > capacity should accrue history penalty.
        """
        acc = accountant_with_usage
        initial_history = acc.history.copy()
        
        # Update history with gain=1.0
        acc.update_history(gain=1.0, decay_factor=1.0)  # No decay for this test
        
        # Edges 0-2 are overused by 1.0 (present=2, capacity=1)
        # Expected increment: gain * overuse = 1.0 * 1.0 = 1.0
        assert acc.history[0] > initial_history[0], "Overused edge should gain history"
        assert acc.history[3] == initial_history[3], "At-capacity edge should not gain history"
        assert acc.history[6] == initial_history[6], "Under-capacity edge should not gain history"

    def test_history_respects_decay_factor(self, accountant_with_usage):
        """Test that history decays before new penalty added.
        
        Formula: history = (history_old * decay) + (gain * overuse)
        """
        acc = accountant_with_usage
        acc.history[0] = 10.0  # Pre-existing history
        
        decay_factor = 0.5
        gain = 2.0
        
        acc.update_history(gain=gain, decay_factor=decay_factor)
        
        # Expected: (10.0 * 0.5) + (2.0 * 1.0) = 5.0 + 2.0 = 7.0
        # (edge 0 has overuse=1.0 because present=2, capacity=1)
        expected = (10.0 * decay_factor) + (gain * 1.0)
        assert np.isclose(acc.history[0], expected, atol=0.01)

    def test_history_cap_enforcement(self, accountant_with_usage, base_costs):
        """Test that history increment is capped at history_cap_multiplier * base_cost.
        
        Prevents runaway history costs that would dominate routing.
        """
        acc = accountant_with_usage
        acc.present_ema[0] = 100.0  # Extreme overuse
        acc.capacity[0] = 1.0
        
        gain = 10.0
        history_cap_multiplier = 5.0
        base_cost = 2.0
        base_costs_arr = np.full(10, base_cost, dtype=np.float32)
        
        acc.update_history(
            gain=gain,
            base_costs=base_costs_arr,
            history_cap_multiplier=history_cap_multiplier,
            decay_factor=1.0
        )
        
        # Without cap: increment = 10.0 * (100.0 - 1.0) = 990.0
        # With cap: increment = min(990.0, 5.0 * 2.0) = 10.0
        max_history_cap = history_cap_multiplier * base_cost
        assert acc.history[0] <= max_history_cap, \
            f"History {acc.history[0]:.2f} should be capped at {max_history_cap}"

    def test_history_uses_present_ema_by_default(self, accountant_with_usage):
        """Test that history uses smoothed present_ema, not raw present.
        
        This prevents bang-bang oscillations in history accumulation.
        """
        acc = accountant_with_usage
        
        # Set different values for present and present_ema
        acc.present[0] = 10.0
        acc.present_ema[0] = 5.0
        acc.capacity[0] = 1.0
        
        acc.update_history(gain=1.0, use_raw_present=False, decay_factor=1.0)
        
        # History should use present_ema (5.0), so overuse = 5.0 - 1.0 = 4.0
        # increment = 1.0 * 4.0 = 4.0
        assert np.isclose(acc.history[0], 4.0, atol=0.01)

    def test_history_can_use_raw_present(self, accountant_with_usage):
        """Test that history can optionally use raw present instead of EMA."""
        acc = accountant_with_usage
        
        acc.present[0] = 10.0
        acc.present_ema[0] = 5.0
        acc.capacity[0] = 1.0
        
        acc.update_history(gain=1.0, use_raw_present=True, decay_factor=1.0)
        
        # History should use raw present (10.0), so overuse = 10.0 - 1.0 = 9.0
        # increment = 1.0 * 9.0 = 9.0
        assert np.isclose(acc.history[0], 9.0, atol=0.01)

    def test_history_decay_reduces_old_penalties(self, small_accountant):
        """Test that decay factor reduces old history penalties over time."""
        acc = small_accountant
        acc.history[0] = 100.0
        
        decay_factor = 0.98
        
        # Update without new overuse (present=0, capacity=1)
        acc.update_history(gain=1.0, decay_factor=decay_factor)
        
        # Expected: 100.0 * 0.98 + 0 = 98.0
        assert np.isclose(acc.history[0], 98.0)
        
        # After 10 iterations
        for _ in range(9):
            acc.update_history(gain=1.0, decay_factor=decay_factor)
        
        # Expected: 100.0 * (0.98^10) ≈ 81.7
        expected = 100.0 * (decay_factor ** 10)
        assert np.isclose(acc.history[0], expected, atol=0.5)


# ============================================================================
# TestCostUpdate - Total cost computation with layer bias
# ============================================================================

class TestCostUpdate:
    """Test update_costs() with base costs, present penalties, and layer bias."""

    def test_update_costs_basic_formula(self, accountant_with_usage, base_costs):
        """Test basic cost formula: base + pres_fac*overuse + hist_weight*history.
        
        This is the core PathFinder cost function.
        """
        acc = accountant_with_usage
        acc.history[0] = 2.0
        
        pres_fac = 1.5
        hist_weight = 1.0
        
        acc.update_costs(base_costs, pres_fac=pres_fac, hist_weight=hist_weight, add_jitter=False)
        
        # Edge 0: overuse = 2.0 - 1.0 = 1.0 (uses present_ema)
        # cost = base(1.0) + pres(1.5*1.0) + hist(1.0*2.0) = 1.0 + 1.5 + 2.0 = 4.5
        # Note: base_cost_weight default is 0.01, so actual base = 1.0 * 0.01 = 0.01
        expected_base = 1.0 * 0.01  # base_cost_weight default
        expected = expected_base + (pres_fac * 1.0) + (hist_weight * 2.0)
        
        assert np.isclose(acc.total_cost[0], expected, atol=0.1), \
            f"Expected {expected:.2f}, got {acc.total_cost[0]:.2f}"

    def test_update_costs_no_overuse_equals_base(self, small_accountant, base_costs):
        """Test that cost equals base_cost when no overuse and no history."""
        acc = small_accountant
        # No usage, no history
        
        acc.update_costs(base_costs, pres_fac=1.0, hist_weight=1.0, add_jitter=False)
        
        # All edges should cost base_cost * base_cost_weight (default 0.01)
        expected = base_costs * 0.01
        assert np.allclose(acc.total_cost, expected, atol=1e-5)

    def test_update_costs_jitter_breaks_ties(self, small_accountant, base_costs):
        """Test that jitter adds small epsilon to break ties.
        
        Prevents oscillation when multiple paths have equal cost.
        """
        acc = small_accountant
        
        acc.update_costs(base_costs, pres_fac=1.0, add_jitter=True)
        
        # Jitter should make consecutive edges have slightly different costs
        assert not np.allclose(acc.total_cost, acc.total_cost[0]), \
            "Jitter should create small cost differences"
        
        # But jitter should be tiny (< 1e-4)
        cost_range = acc.total_cost.max() - acc.total_cost.min()
        assert cost_range < 1e-3, f"Jitter range too large: {cost_range}"

    def test_update_costs_via_cost_multiplier(self, small_accountant):
        """Test that via_cost_multiplier scales base costs (for late-stage annealing)."""
        acc = small_accountant
        base_costs = np.array([1.0] * 10, dtype=np.float32)
        
        via_multiplier = 2.0
        
        acc.update_costs(base_costs, pres_fac=0.0, hist_weight=0.0, 
                        via_cost_multiplier=via_multiplier, add_jitter=False)
        
        # cost = base * via_mult * base_weight = 1.0 * 2.0 * 0.01 = 0.02
        expected = base_costs * via_multiplier * 0.01
        assert np.allclose(acc.total_cost, expected, atol=1e-5)

    def test_update_costs_layer_bias_applied_to_base(self, small_accountant):
        """Test that layer bias correctly scales base costs for H/V edges.
        
        Layer bias enables rebalancing congested layers by making them cheaper
        in the base cost term (encourages use) and more expensive in present
        term (discourages overuse).
        """
        acc = small_accountant
        base_costs = np.array([1.0] * 10, dtype=np.float32)
        
        # Set up layers and bias
        edge_layer = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        layer_bias = np.array([0.5, 1.0, 1.5, 1.0, 1.0], dtype=np.float32)  # 5 layers
        edge_kind = np.zeros(10, dtype=np.int32)  # All H/V edges (not vias)
        
        acc.update_costs(
            base_costs, 
            pres_fac=0.0, 
            hist_weight=0.0,
            add_jitter=False,
            edge_layer=edge_layer,
            layer_bias_per_layer=layer_bias,
            edge_kind=edge_kind
        )
        
        # Edges 0-1 (layer 0): bias=0.5 → cost = 1.0 * 0.5 * 0.01 = 0.005
        # Edges 2-3 (layer 1): bias=1.0 → cost = 1.0 * 1.0 * 0.01 = 0.01
        # Edges 4-5 (layer 2): bias=1.5 → cost = 1.0 * 1.5 * 0.01 = 0.015
        
        assert np.isclose(acc.total_cost[0], 0.005, atol=1e-5), \
            f"Layer 0 edge should have bias 0.5, got cost {acc.total_cost[0]}"
        assert np.isclose(acc.total_cost[2], 0.01, atol=1e-5), \
            f"Layer 1 edge should have bias 1.0, got cost {acc.total_cost[2]}"
        assert np.isclose(acc.total_cost[4], 0.015, atol=1e-5), \
            f"Layer 2 edge should have bias 1.5, got cost {acc.total_cost[4]}"

    def test_update_costs_layer_bias_not_applied_to_vias(self, small_accountant):
        """Test that layer bias is NOT applied to via edges.
        
        Vias should maintain uniform cost regardless of layer bias to prevent
        weird via-avoidance behavior.
        """
        acc = small_accountant
        base_costs = np.array([1.0] * 10, dtype=np.float32)
        
        edge_layer = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        layer_bias = np.array([0.5, 1.0, 1.5, 1.0, 1.0], dtype=np.float32)
        edge_kind = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)  # Alternating H/V and via
        
        acc.update_costs(
            base_costs,
            pres_fac=0.0,
            hist_weight=0.0,
            add_jitter=False,
            edge_layer=edge_layer,
            layer_bias_per_layer=layer_bias,
            edge_kind=edge_kind
        )
        
        # Via edges (odd indices) should all have same cost (bias=1.0)
        via_costs = acc.total_cost[1::2]  # indices 1, 3, 5, 7, 9
        assert np.allclose(via_costs, via_costs[0], atol=1e-5), \
            f"All via costs should be equal, got {via_costs}"


# ============================================================================
# TestOveruseComputation - Overuse detection
# ============================================================================

class TestOveruseComputation:
    """Test compute_overuse() edge congestion detection."""

    def test_compute_overuse_counts_overused_edges(self, accountant_with_usage):
        """Test that compute_overuse() correctly counts overused edges."""
        acc = accountant_with_usage
        
        over_sum, over_count = acc.compute_overuse()
        
        # Edges 0-2 are overused by 1.0 each (present=2, capacity=1)
        # Total overuse = 3 * 1.0 = 3.0
        # Overused edge count = 3
        assert over_sum == 3, f"Expected overuse sum 3, got {over_sum}"
        assert over_count == 3, f"Expected 3 overused edges, got {over_count}"

    def test_compute_overuse_zero_when_no_congestion(self, small_accountant):
        """Test that compute_overuse() returns (0, 0) with no overuse."""
        acc = small_accountant
        # No usage
        
        over_sum, over_count = acc.compute_overuse()
        
        assert over_sum == 0
        assert over_count == 0

    def test_compute_overuse_ignores_under_capacity_edges(self, accountant_with_usage):
        """Test that compute_overuse() only counts edges exceeding capacity."""
        acc = accountant_with_usage
        
        # Edges 3-5 at capacity (usage=1, capacity=1) → no overuse
        # Edges 6-9 under capacity (usage=0.5, capacity=1) → no overuse
        
        over_sum, over_count = acc.compute_overuse()
        
        # Only edges 0-2 should count
        assert over_count == 3


# ============================================================================
# TestGPUConsistency - CPU/GPU parity
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUConsistency:
    """Test that GPU and CPU implementations produce identical results."""

    def test_present_ema_cpu_gpu_parity(self):
        """Test that EMA computation is identical on CPU and GPU."""
        from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
        
        acc_cpu = EdgeAccountant(num_edges=10, use_gpu=False)
        acc_gpu = EdgeAccountant(num_edges=10, use_gpu=True)
        
        # Set same initial state
        acc_cpu.present[:] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        acc_gpu.present[:] = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=cp.float32)
        
        acc_cpu.update_present_ema(beta=0.6)
        acc_gpu.update_present_ema(beta=0.6)
        
        # Compare results
        cpu_result = acc_cpu.present_ema
        gpu_result = acc_gpu.present_ema.get()
        
        assert np.allclose(cpu_result, gpu_result, atol=1e-5), \
            f"CPU/GPU EMA mismatch: {cpu_result} vs {gpu_result}"

    def test_history_update_cpu_gpu_parity(self):
        """Test that history update is identical on CPU and GPU."""
        from orthoroute.algorithms.manhattan.unified_pathfinder import EdgeAccountant
        
        acc_cpu = EdgeAccountant(num_edges=10, use_gpu=False)
        acc_gpu = EdgeAccountant(num_edges=10, use_gpu=True)
        
        # Set same initial state with overuse
        acc_cpu.present_ema[:] = np.array([2.0] * 10, dtype=np.float32)
        acc_gpu.present_ema[:] = cp.array([2.0] * 10, dtype=cp.float32)
        
        acc_cpu.update_history(gain=1.0, decay_factor=0.98)
        acc_gpu.update_history(gain=1.0, decay_factor=0.98)
        
        cpu_result = acc_cpu.history
        gpu_result = acc_gpu.history.get()
        
        assert np.allclose(cpu_result, gpu_result, atol=1e-4), \
            f"CPU/GPU history mismatch"
