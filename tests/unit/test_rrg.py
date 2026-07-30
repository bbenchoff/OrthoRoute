"""Unit tests for rrg.py - Routing Resource Graph data structures.

Tests RRGNode, RRGEdge, and RoutingConfig to ensure correct capacity tracking,
utilization calculation, and PathFinder cost computation with congestion.
"""
import pytest
import math
from orthoroute.algorithms.manhattan.rrg import (
    RRGNode,
    RRGEdge,
    RoutingConfig,
    NodeType,
    EdgeType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_node():
    """Basic routing node with default capacity."""
    return RRGNode(
        id="node_1",
        node_type=NodeType.RAIL,
        x=10.0,
        y=20.0,
        layer=1,
        capacity=1,
        usage=0,
        track_index=0
    )


@pytest.fixture
def high_capacity_node():
    """Node with capacity > 1 for multi-track testing."""
    return RRGNode(
        id="node_multi",
        node_type=NodeType.BUS,
        x=5.0,
        y=15.0,
        layer=2,
        capacity=3,
        usage=0,
        track_index=1
    )


@pytest.fixture
def basic_edge():
    """Basic routing edge connecting two nodes."""
    return RRGEdge(
        id="edge_1",
        edge_type=EdgeType.TRACK,
        from_node="node_1",
        to_node="node_2",
        length_mm=5.0,
        capacity=1,
        usage=0,
        base_cost=5.0,
        history_cost=0.0
    )


@pytest.fixture
def default_config():
    """Default PathFinder configuration."""
    return RoutingConfig()


# ============================================================================
# TestRRGNode - Node capacity and utilization
# ============================================================================

class TestRRGNode:
    """Test RRGNode creation, availability, and utilization calculation."""

    def test_rrg_node_creation_with_defaults(self, basic_node):
        """Test that RRGNode constructs with valid parameters.
        
        Validates core data structure for PathFinder graph nodes.
        """
        assert basic_node.id == "node_1"
        assert basic_node.node_type == NodeType.RAIL
        assert basic_node.x == 10.0
        assert basic_node.y == 20.0
        assert basic_node.layer == 1
        assert basic_node.capacity == 1
        assert basic_node.usage == 0
        assert basic_node.track_index == 0

    def test_is_available_when_under_capacity(self, basic_node):
        """Test that is_available() returns True when usage < capacity.
        
        Critical for PathFinder expansion: only available nodes can be added to paths.
        """
        basic_node.usage = 0
        assert basic_node.is_available() is True
        
    def test_is_available_at_capacity(self, basic_node):
        """Test that is_available() returns False when usage == capacity.
        
        Prevents over-subscription during negotiated congestion routing.
        """
        basic_node.usage = 1  # capacity = 1
        assert basic_node.is_available() is False

    def test_is_available_over_capacity(self, high_capacity_node):
        """Test that is_available() returns False when usage > capacity.
        
        Handles transient over-subscription during PathFinder iterations.
        """
        high_capacity_node.usage = 4  # capacity = 3
        assert high_capacity_node.is_available() is False

    def test_utilization_zero_usage(self, basic_node):
        """Test that utilization() returns 0.0 for unused node."""
        basic_node.usage = 0
        assert basic_node.utilization() == 0.0

    def test_utilization_partial_capacity(self, high_capacity_node):
        """Test that utilization() computes correct ratio for partial usage."""
        high_capacity_node.usage = 2  # 2 of 3 capacity
        expected = 2.0 / 3.0
        assert math.isclose(high_capacity_node.utilization(), expected, abs_tol=1e-9)

    def test_utilization_at_capacity(self, basic_node):
        """Test that utilization() returns 1.0 when fully utilized."""
        basic_node.usage = 1  # capacity = 1
        assert basic_node.utilization() == 1.0

    def test_utilization_over_capacity(self, high_capacity_node):
        """Test that utilization() exceeds 1.0 when oversubscribed.
        
        Needed for PathFinder congestion penalty: (utilization)^alpha term.
        """
        high_capacity_node.usage = 6  # 6 of 3 capacity
        assert high_capacity_node.utilization() == 2.0

    def test_utilization_zero_capacity_node(self):
        """Test that utilization() returns 0.0 for zero-capacity node.
        
        Prevents division by zero for obstacle nodes.
        """
        node = RRGNode(
            id="obstacle",
            node_type=NodeType.SWITCH,
            x=0.0,
            y=0.0,
            layer=0,
            capacity=0,
            usage=0
        )
        assert node.utilization() == 0.0


# ============================================================================
# TestRRGEdge - Edge cost calculation with congestion
# ============================================================================

class TestRRGEdge:
    """Test RRGEdge cost computation with PathFinder congestion penalties."""

    def test_rrg_edge_creation_with_defaults(self, basic_edge):
        """Test that RRGEdge constructs with valid parameters.
        
        Validates core data structure for PathFinder graph edges.
        """
        assert basic_edge.id == "edge_1"
        assert basic_edge.edge_type == EdgeType.TRACK
        assert basic_edge.from_node == "node_1"
        assert basic_edge.to_node == "node_2"
        assert basic_edge.length_mm == 5.0
        assert basic_edge.capacity == 1
        assert basic_edge.usage == 0
        assert basic_edge.base_cost == 5.0
        assert basic_edge.history_cost == 0.0

    def test_is_available_edge_under_capacity(self, basic_edge):
        """Test that edge is_available() returns True when usage < capacity."""
        basic_edge.usage = 0
        assert basic_edge.is_available() is True

    def test_is_available_edge_at_capacity(self, basic_edge):
        """Test that edge is_available() returns False at capacity."""
        basic_edge.usage = 1  # capacity = 1
        assert basic_edge.is_available() is False

    def test_edge_utilization_calculation(self, basic_edge):
        """Test that edge utilization() computes correct ratio."""
        basic_edge.usage = 0
        assert basic_edge.utilization() == 0.0
        
        basic_edge.usage = 1  # capacity = 1
        assert basic_edge.utilization() == 1.0

    def test_edge_cost_no_congestion(self, basic_edge):
        """Test that current_cost() equals base_cost when no usage.
        
        Baseline pathfinding without congestion penalties.
        """
        basic_edge.usage = 0
        basic_edge.base_cost = 10.0
        basic_edge.history_cost = 0.0
        
        cost = basic_edge.current_cost(pres_fac=1.0, alpha=2.0)
        
        # cost = base_cost * (1 + 0) + 0 = base_cost
        assert math.isclose(cost, 10.0, abs_tol=1e-9)

    def test_edge_cost_with_congestion_no_history(self, basic_edge):
        """Test that current_cost() applies present penalty with congestion.
        
        PathFinder present penalty: base_cost * (1 + pres_fac * (util)^alpha)
        """
        basic_edge.usage = 1  # capacity = 1, so utilization = 1.0
        basic_edge.base_cost = 10.0
        basic_edge.history_cost = 0.0
        
        pres_fac = 0.5
        alpha = 2.0
        
        cost = basic_edge.current_cost(pres_fac=pres_fac, alpha=alpha)
        
        # present_penalty = 0.5 * (1.0)^2.0 = 0.5
        # cost = 10.0 * (1 + 0.5) + 0 = 15.0
        expected = 15.0
        assert math.isclose(cost, expected, abs_tol=1e-9)

    def test_edge_cost_with_history_penalty(self, basic_edge):
        """Test that current_cost() adds history cost.
        
        History cost accumulates across iterations for persistent overuse.
        """
        basic_edge.usage = 0
        basic_edge.base_cost = 10.0
        basic_edge.history_cost = 5.0  # Accumulated penalty
        
        cost = basic_edge.current_cost(pres_fac=1.0, alpha=2.0)
        
        # cost = 10.0 * (1 + 0) + 5.0 = 15.0
        expected = 15.0
        assert math.isclose(cost, expected, abs_tol=1e-9)

    def test_edge_cost_with_both_penalties(self, basic_edge):
        """Test that current_cost() combines present and history penalties.
        
        Full PathFinder cost formula validation.
        """
        basic_edge.usage = 1  # utilization = 1.0
        basic_edge.base_cost = 10.0
        basic_edge.history_cost = 3.0
        
        pres_fac = 0.5
        alpha = 2.0
        
        cost = basic_edge.current_cost(pres_fac=pres_fac, alpha=alpha)
        
        # present_penalty = 0.5 * (1.0)^2.0 = 0.5
        # cost = 10.0 * (1 + 0.5) + 3.0 = 18.0
        expected = 18.0
        assert math.isclose(cost, expected, abs_tol=1e-9)

    def test_edge_cost_with_partial_utilization(self):
        """Test current_cost() with fractional utilization.
        
        Validates exponentiation for multi-capacity edges.
        """
        edge = RRGEdge(
            id="multi_edge",
            edge_type=EdgeType.TRACK,
            from_node="n1",
            to_node="n2",
            length_mm=5.0,
            capacity=4,
            usage=2,  # utilization = 0.5
            base_cost=8.0,
            history_cost=0.0
        )
        
        pres_fac = 1.0
        alpha = 2.0
        
        cost = edge.current_cost(pres_fac=pres_fac, alpha=alpha)
        
        # utilization = 2/4 = 0.5
        # present_penalty = 1.0 * (0.5)^2.0 = 0.25
        # cost = 8.0 * (1 + 0.25) + 0 = 10.0
        expected = 10.0
        assert math.isclose(cost, expected, abs_tol=1e-9)

    def test_edge_cost_zero_capacity_prevents_division_by_zero(self):
        """Test that zero capacity edge handles utilization gracefully.
        
        Prevents crashes for obstacle edges.
        """
        edge = RRGEdge(
            id="obstacle_edge",
            edge_type=EdgeType.TRACK,
            from_node="n1",
            to_node="n2",
            length_mm=1.0,
            capacity=0,
            usage=0,
            base_cost=999.0,
            history_cost=0.0
        )
        
        # Should not crash - utilization() returns 0.0
        cost = edge.current_cost(pres_fac=1.0, alpha=2.0)
        assert math.isclose(cost, 999.0, abs_tol=1e-9)


# ============================================================================
# TestRoutingConfig - Configuration validation
# ============================================================================

class TestRoutingConfig:
    """Test RoutingConfig defaults and parameter validation."""

    def test_routing_config_default_values(self, default_config):
        """Test that RoutingConfig constructs with documented defaults.
        
        Ensures consistent PathFinder behavior across runs.
        """
        assert default_config.grid_pitch == 0.4
        assert default_config.track_width == 0.0889
        assert default_config.clearance == 0.0889
        assert default_config.via_diameter == 0.25
        assert default_config.via_drill == 0.15
        assert default_config.k_length == 1.0
        assert default_config.k_via == 10.0
        assert default_config.k_bend == 2.0
        assert default_config.max_iterations == 50
        assert default_config.pres_fac_init == 0.5
        assert default_config.pres_fac_mult == 1.4
        assert default_config.hist_cost_step == 1.0
        assert default_config.alpha == 2.0

    def test_routing_config_custom_values(self):
        """Test that RoutingConfig accepts custom parameters.
        
        Allows parameter tuning for different board densities.
        """
        config = RoutingConfig(
            grid_pitch=0.5,
            max_iterations=100,
            pres_fac_init=1.0,
            alpha=3.0
        )
        
        assert config.grid_pitch == 0.5
        assert config.max_iterations == 100
        assert config.pres_fac_init == 1.0
        assert config.alpha == 3.0
        # Other params should still have defaults
        assert config.k_via == 10.0

    def test_routing_config_pathfinder_multiplier(self):
        """Test pres_fac_mult enables exponential congestion escalation.
        
        pres_fac *= pres_fac_mult each iteration → forces nets to negotiate.
        """
        config = RoutingConfig(pres_fac_init=0.5, pres_fac_mult=1.4)
        
        # Simulate 3 iterations
        pres_fac = config.pres_fac_init
        pres_fac *= config.pres_fac_mult  # Iteration 2: 0.7
        pres_fac *= config.pres_fac_mult  # Iteration 3: 0.98
        
        assert math.isclose(pres_fac, 0.98, abs_tol=0.01)

    def test_routing_config_alpha_exponent_effect(self):
        """Test that alpha parameter controls congestion penalty steepness.
        
        Higher alpha → more aggressive avoidance of overused edges.
        """
        config1 = RoutingConfig(alpha=1.0)
        config2 = RoutingConfig(alpha=3.0)
        
        # Simulate 50% utilization penalty
        util = 0.5
        penalty1 = util ** config1.alpha  # 0.5
        penalty2 = util ** config2.alpha  # 0.125
        
        assert penalty1 > penalty2  # Lower alpha → higher relative penalty at partial util
        
        # Simulate 100% utilization penalty
        util = 1.0
        penalty1_full = util ** config1.alpha  # 1.0
        penalty2_full = util ** config2.alpha  # 1.0
        
        assert penalty1_full == penalty2_full  # At full util, alpha doesn't matter
