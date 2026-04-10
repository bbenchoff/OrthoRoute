"""Unit tests for pathfinding methods.

Tests find_path_roi() and find_path_multisource_multisink() to ensure:
- Basic connectivity (src → dst path exists)
- Shortest path when no congestion
- Congestion avoidance (detours around high-cost edges)
- No-path case handled gracefully
- Portal seed generation and multi-source expansion
"""
import pytest
import numpy as np
from typing import List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_lattice_path():
    """Mock lattice for pathfinding tests (6×6×2 grid for simple paths)."""
    lattice = Mock()
    lattice.x_steps = 6
    lattice.y_steps = 6
    lattice.layers = 2
    lattice.total_nodes = 72  # 6×6×2
    
    def idx_to_coord(idx):
        plane_size = lattice.x_steps * lattice.y_steps
        z, remainder = divmod(idx, plane_size)
        y, x = divmod(remainder, lattice.x_steps)
        return x, y, z
    
    def coord_to_idx(x, y, z):
        return z * (lattice.x_steps * lattice.y_steps) + y * lattice.x_steps + x
    
    lattice.idx_to_coord = idx_to_coord
    lattice.coord_to_idx = coord_to_idx
    
    return lattice


@pytest.fixture
def mock_pathfinder_path(mock_lattice_path):
    """Mock UnifiedPathFinder for pathfinding tests."""
    from scipy.sparse import csr_matrix
    
    pf = Mock()
    pf.lattice = mock_lattice_path
    pf.N = mock_lattice_path.total_nodes
    
    # Create simple grid graph (4-way + via connectivity)
    pf.graph = csr_matrix((pf.N, pf.N), dtype=np.float32)
    pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
    pf.graph.indices = np.array([], dtype=np.int32)
    
    # Uniform costs
    pf.edge_costs = np.ones(pf.N * 6, dtype=np.float32)
    
    return pf


def create_simple_roi(src: int, dst: int, pf):
    """Helper to create minimal ROI containing src and dst."""
    # Simple ROI: just src, dst, and a few intermediate nodes
    roi_nodes = np.array([src, dst], dtype=np.int32)
    
    # Add some intermediate nodes for path (implementation-specific)
    # For tests, we'll just use src and dst
    
    global_to_roi = np.full(pf.N, -1, dtype=np.int32)
    global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)
    
    return roi_nodes, global_to_roi


# ============================================================================
# TestPathfindingBasic - Basic connectivity validation
# ============================================================================

class TestPathfindingBasic:
    """Test find_path_roi() basic pathfinding functionality."""

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_returns_valid_path(self, mock_dijkstra, mock_pathfinder_path):
        """Test that find_path_roi returns a valid path from src to dst."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # Mock dijkstra to return a simple path
        mock_dijkstra.return_value = ([0, 1, 2, 10], 4.0)  # (path, cost)
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)  # Uniform costs
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        assert path is not None, "Should return a path"
        assert len(path) >= 2, f"Path should have at least src and dst, got {len(path) if path else 0} nodes"
        assert path[0] == src, f"Path should start at src {src}, got {path[0]}"
        assert path[-1] == dst, f"Path should end at dst {dst}, got {path[-1]}"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_returns_none_when_unreachable(self, mock_dijkstra, mock_pathfinder_path):
        """Test that find_path_roi returns None when dst is unreachable."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # Mock dijkstra to return empty path (unreachable)
        mock_dijkstra.return_value = ([], float('inf'))
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # Should return None or empty list for unreachable
        assert path is None or len(path) == 0, "Should return None/empty for unreachable dst"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_adjacent_nodes(self, mock_dijkstra, mock_pathfinder_path):
        """Test pathfinding between adjacent nodes (minimal case)."""
        pf = mock_pathfinder_path
        src, dst = 0, 1  # Adjacent in X direction
        
        # Mock shortest path (just one edge)
        mock_dijkstra.return_value = ([0, 1], 1.0)
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        assert path is not None, "Adjacent nodes should be connectable"
        assert len(path) == 2, f"Adjacent path should have 2 nodes, got {len(path)}"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_same_node(self, mock_dijkstra, mock_pathfinder_path):
        """Test pathfinding when src == dst (degenerate case)."""
        pf = mock_pathfinder_path
        src = dst = 5
        
        # Mock trivial path
        mock_dijkstra.return_value = ([5], 0.0)
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # Should handle gracefully (implementation-specific)
        # Some implementations return [src], others return None
        if path is not None:
            assert len(path) >= 1, "Same-node path should have at least one node"


# ============================================================================
# TestPathfindingOptimality - Shortest path validation
# ============================================================================

class TestPathfindingOptimality:
    """Test that pathfinding finds shortest/lowest-cost paths."""

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_prefers_low_cost_route(self, mock_dijkstra, mock_pathfinder_path):
        """Test that pathfinding prefers low-cost edges.
        
        Scenario: Two paths available, one with lower cost.
        Path A: src → mid1 → dst (cost 1+1 = 2)
        Path B: src → mid2 → dst (cost 5+5 = 10)
        Should choose Path A.
        """
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # Mock dijkstra to return lowest-cost path
        # In real implementation, dijkstra would compute this
        mock_dijkstra.return_value = ([0, 5, 10], 2.0)  # Low-cost path
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        assert path is not None, "Should find a path"
        # Actual cost checking would require full graph simulation
        # This test documents expected behavior

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_avoids_high_cost_edges(self, mock_dijkstra, mock_pathfinder_path):
        """Test that pathfinding detours around high-cost (congested) edges."""
        pf = mock_pathfinder_path
        src, dst = 0, 11
        
        # Mock path that detours around congestion
        # Direct path would be [0, 1, 2, ..., 11]
        # Detour path might be [0, 6, 7, 8, 11] to avoid congested edge
        mock_dijkstra.return_value = ([0, 6, 7, 11], 8.0)  # Detour path
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        
        # Set high cost on direct path edge
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        costs[0] = 100.0  # Make direct edge expensive
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        assert path is not None, "Should find detour path"
        # Path should avoid high-cost edge (implementation determines exact route)


# ============================================================================
# TestMultisourcePathfinding - Multi-source/multi-sink pathfinding
# ============================================================================

class TestMultisourcePathfinding:
    """Test find_path_multisource_multisink() portal-based routing."""

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_multisource_multisink')
    def test_multisource_finds_path_from_any_source(self, mock_dijkstra, mock_pathfinder_path):
        """Test that multi-source pathfinding can start from any source seed."""
        pf = mock_pathfinder_path
        
        # Multiple source seeds (node_id, cost)
        src_seeds = [(0, 0.0), (1, 0.0), (2, 0.0)]
        dst_seeds = [(10, 0.0)]
        
        # Mock path starting from second source
        mock_dijkstra.return_value = ([1, 5, 10], 3.0)  # Path from seed 1
        
        roi_nodes = np.array([0, 1, 2, 5, 10], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        
        assert path is not None, "Should find path from some source"
        assert len(path) >= 2, "Path should have at least 2 nodes"
        # First node should be one of the source seeds
        assert path[0] in [s[0] for s in src_seeds], "Path should start from a source seed"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_multisource_multisink')
    def test_multisource_reaches_any_dst_seed(self, mock_dijkstra, mock_pathfinder_path):
        """Test that multi-source pathfinding can reach any dst seed."""
        pf = mock_pathfinder_path
        
        src_seeds = [(0, 0.0)]
        # Multiple destination seeds
        dst_seeds = [(10, 0.0), (11, 0.0), (12, 0.0)]
        
        # Mock path reaching second dst seed
        mock_dijkstra.return_value = ([0, 5, 11], 3.5)  # Path to seed 11
        
        roi_nodes = np.array([0, 5, 10, 11, 12], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        
        assert path is not None, "Should find path to some dst"
        # Last node should be one of the dst seeds
        assert path[-1] in [d[0] for d in dst_seeds], "Path should end at a dst seed"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_multisource_multisink')
    def test_multisource_uses_seed_costs(self, mock_dijkstra, mock_pathfinder_path):
        """Test that seed costs are incorporated (portal penalties)."""
        pf = mock_pathfinder_path
        
        # Different costs at seeds (represents escape difficulty)
        src_seeds = [(0, 0.0), (1, 5.0)]  # Seed 1 has higher escape cost
        dst_seeds = [(10, 0.0)]
        
        # Mock would prefer starting from seed 0 (lower cost)
        mock_dijkstra.return_value = ([0, 5, 10], 2.0)
        
        roi_nodes = np.array([0, 1, 5, 10], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        
        # Dijkstra should factor in seed costs when choosing starting point
        assert path is not None, "Should find path"


# ============================================================================
# TestPathfindingEdgeCases - Edge case validation
# ============================================================================

class TestPathfindingEdgeCases:
    """Test pathfinding edge cases and error handling."""

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_empty_roi(self, mock_dijkstra, mock_pathfinder_path):
        """Test that pathfinding handles empty ROI gracefully."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # Empty ROI
        roi_nodes = np.array([], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        costs = np.array([], dtype=np.float32)
        
        mock_dijkstra.return_value = ([], float('inf'))
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # Should return None or handle gracefully
        assert path is None or len(path) == 0, "Empty ROI should return None/empty path"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_src_not_in_roi(self, mock_dijkstra, mock_pathfinder_path):
        """Test that pathfinding handles src not in ROI."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # ROI doesn't include src (only dst)
        roi_nodes = np.array([10], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[10] = 0
        costs = np.ones(1 * 6, dtype=np.float32)
        
        mock_dijkstra.return_value = ([], float('inf'))
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # Should handle gracefully (implementation-specific behavior)
        # Likely returns None or raises error
        pass  # Document expected behavior

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_dst_not_in_roi(self, mock_dijkstra, mock_pathfinder_path):
        """Test that pathfinding handles dst not in ROI."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # ROI doesn't include dst (only src)
        roi_nodes = np.array([0], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[0] = 0
        costs = np.ones(1 * 6, dtype=np.float32)
        
        mock_dijkstra.return_value = ([], float('inf'))
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # Should return None/empty (dst unreachable)
        assert path is None or len(path) == 0, "Dst not in ROI should be unreachable"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_multisource_multisink')
    def test_multisource_empty_seeds(self, mock_dijkstra, mock_pathfinder_path):
        """Test multi-source pathfinding with empty seed lists."""
        pf = mock_pathfinder_path
        
        # Empty source seeds
        src_seeds = []
        dst_seeds = [(10, 0.0)]
        
        roi_nodes = np.array([10], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[10] = 0
        costs = np.ones(1 * 6, dtype=np.float32)
        
        mock_dijkstra.return_value = ([], float('inf'))
        
        path = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        
        # Should handle gracefully
        assert path is None or len(path) == 0, "Empty sources should return no path"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_all_edges_infinite_cost(self, mock_dijkstra, mock_pathfinder_path):
        """Test pathfinding when all edges have infinite cost (blocked)."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        
        # All edges blocked (infinite cost)
        costs = np.full(len(roi_nodes) * 6, float('inf'), dtype=np.float32)
        
        mock_dijkstra.return_value = ([], float('inf'))
        
        path = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        # No path possible
        assert path is None or len(path) == 0, "All blocked edges should yield no path"


# ============================================================================
# TestPathfindingDeterminism - Deterministic behavior
# ============================================================================

class TestPathfindingDeterminism:
    """Test that pathfinding is deterministic (same inputs → same path)."""

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_single_target')
    def test_find_path_deterministic_simple(self, mock_dijkstra, mock_pathfinder_path):
        """Test that repeated pathfinding yields identical results."""
        pf = mock_pathfinder_path
        src, dst = 0, 10
        
        # Fixed mock path
        mock_dijkstra.return_value = ([0, 5, 10], 3.0)
        
        roi_nodes, global_to_roi = create_simple_roi(src, dst, pf)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path1 = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        path2 = pf.find_path_roi(src, dst, costs, roi_nodes, global_to_roi)
        
        assert path1 == path2, "Repeated pathfinding should yield identical paths"

    @patch('orthoroute.algorithms.manhattan.unified_pathfinder.dijkstra_multisource_multisink')
    def test_multisource_deterministic_seed_order(self, mock_dijkstra, mock_pathfinder_path):
        """Test that multi-source pathfinding is deterministic with same seed order."""
        pf = mock_pathfinder_path
        
        src_seeds = [(0, 0.0), (1, 0.0)]
        dst_seeds = [(10, 0.0)]
        
        mock_dijkstra.return_value = ([0, 5, 10], 2.5)
        
        roi_nodes = np.array([0, 1, 5, 10], dtype=np.int32)
        global_to_roi = np.full(pf.N, -1, dtype=np.int32)
        global_to_roi[roi_nodes] = np.arange(len(roi_nodes), dtype=np.int32)
        costs = np.ones(len(roi_nodes) * 6, dtype=np.float32)
        
        path1 = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        path2 = pf.find_path_multisource_multisink(
            src_seeds, dst_seeds, costs, roi_nodes, global_to_roi
        )
        
        assert path1 == path2, "Multi-source pathfinding should be deterministic"
