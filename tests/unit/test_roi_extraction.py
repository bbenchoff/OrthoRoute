"""Unit tests for ROI (Region of Interest) extraction methods.

Tests geometric bounding box and BFS-based ROI extraction to ensure:
- Src/dst nodes are always included in ROI
- Portal seeds are preserved
- ROI size respects max_nodes budget
- Connectivity is maintained after truncation
- Deterministic behavior (same inputs → same ROI)

NOTE: These are SPECIFICATION TESTS using mocks to document expected behavior.
ROI extraction methods are complex and require full PathFinder initialization.
For actual coverage, see integration tests that use real PathFinder instances.
These tests serve as executable documentation and interface contracts.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_lattice_small():
    """Mock lattice for a small 10×10×4 grid."""
    lattice = Mock()
    lattice.x_steps = 10
    lattice.y_steps = 10
    lattice.layers = 4
    lattice.total_nodes = 400  # 10×10×4
    
    def idx_to_coord(idx):
        """Convert flat index to (x, y, z) coordinates."""
        plane_size = lattice.x_steps * lattice.y_steps
        z, remainder = divmod(idx, plane_size)
        y, x = divmod(remainder, lattice.x_steps)
        return x, y, z
    
    lattice.idx_to_coord = idx_to_coord
    return lattice


@pytest.fixture
def mock_pathfinder_small(mock_lattice_small):
    """Mock UnifiedPathFinder with small grid."""
    # Note: ROI extraction methods are tested at integration level
    # These tests document expected behavior but may require real PathFinder instance
    pf = Mock()
    pf.lattice = mock_lattice_small
    pf.N = mock_lattice_small.total_nodes
    pf.max_roi_nodes = 200  # Smaller budget for testing truncation
    
    # Create minimal graph structure (CSR format)
    from scipy.sparse import csr_matrix
    pf.graph = csr_matrix((pf.N, pf.N), dtype=np.float32)
    pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
    pf.graph.indices = np.array([], dtype=np.int32)
    
    return pf


@pytest.fixture
def basic_nodes():
    """Basic src/dst node indices for a 10×10×4 grid."""
    # Src at (1, 1, 1) = layer 1, (y=1, x=1) = 1*100 + 1*10 + 1 = 111
    # Dst at (8, 8, 2) = layer 2, (y=8, x=8) = 2*100 + 8*10 + 8 = 288
    return {"src": 111, "dst": 288}


# ============================================================================
# TestROIGeometric - Geometric bounding box extraction
# ============================================================================

class TestROIGeometric:
    """Test extract_roi_geometric() bounding box approach."""

    def test_roi_geometric_includes_src_and_dst(self, mock_pathfinder_small, basic_nodes):
        """Test that geometric ROI always includes src and dst nodes."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(src, dst, corridor_buffer=2)
        
        assert src in roi_nodes, f"Src {src} must be in ROI"
        assert dst in roi_nodes, f"Dst {dst} must be in ROI"
        assert global_to_roi[src] >= 0, "Src must have valid ROI mapping"
        assert global_to_roi[dst] >= 0, "Dst must have valid ROI mapping"

    def test_roi_geometric_creates_l_shaped_corridor(self, mock_pathfinder_small, basic_nodes):
        """Test that geometric ROI creates L-shaped corridor between src and dst.
        
        L-corridor should include both possible paths:
        - Horizontal first: src → (dst.x, src.y) → dst
        - Vertical first: src → (src.x, dst.y) → dst
        """
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi_nodes, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=1, layer_margin=1)
        
        # Verify ROI is substantial (not just src+dst)
        assert len(roi_nodes) > 50, f"L-corridor should have many nodes, got {len(roi_nodes)}"
        
        # Get coordinates
        src_x, src_y, src_z = pf.lattice.idx_to_coord(src)
        dst_x, dst_y, dst_z = pf.lattice.idx_to_coord(dst)
        
        # Check that intermediate waypoint coordinates appear in ROI
        # (dst.x, src.y) should be in L-path 1
        # (src.x, dst.y) should be in L-path 2
        roi_coords = [pf.lattice.idx_to_coord(int(n)) for n in roi_nodes]
        
        # Check for presence of waypoint regions (not exact nodes, just general coverage)
        has_horizontal_first = any(abs(x - dst_x) <= 2 and abs(y - src_y) <= 2 for x, y, z in roi_coords)
        has_vertical_first = any(abs(x - src_x) <= 2 and abs(y - dst_y) <= 2 for x, y, z in roi_coords)
        
        assert has_horizontal_first or has_vertical_first, "L-corridor waypoints should be present"

    def test_roi_geometric_respects_corridor_buffer(self, mock_pathfinder_small):
        """Test that corridor_buffer parameter controls ROI width."""
        pf = mock_pathfinder_small
        src, dst = 111, 288
        
        # Small buffer
        roi_small, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=1, layer_margin=1)
        
        # Large buffer
        roi_large, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=5, layer_margin=1)
        
        # Larger buffer should create more nodes
        assert len(roi_large) > len(roi_small), \
            f"Larger buffer should create bigger ROI: small={len(roi_small)}, large={len(roi_large)}"

    def test_roi_geometric_respects_layer_margin(self, mock_pathfinder_small):
        """Test that layer_margin parameter controls vertical extent."""
        pf = mock_pathfinder_small
        src, dst = 111, 288  # Different layers (1 and 2)
        
        # Small layer margin
        roi_small, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=2, layer_margin=0)
        
        # Large layer margin
        roi_large, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=2, layer_margin=2)
        
        # Larger margin should include more layers
        assert len(roi_large) >= len(roi_small), \
            f"Larger layer margin should include more nodes: small={len(roi_small)}, large={len(roi_large)}"

    def test_roi_geometric_includes_portal_seeds(self, mock_pathfinder_small):
        """Test that portal seeds are preserved in ROI."""
        pf = mock_pathfinder_small
        src, dst = 111, 288
        
        # Add portal seeds at random locations
        portal_seeds = [(150, 0.5), (250, 0.5)]  # (node_id, cost)
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(
            src, dst, corridor_buffer=2, layer_margin=1, portal_seeds=portal_seeds
        )
        
        # Verify all portal seeds are in ROI
        for node_id, _ in portal_seeds:
            assert node_id in roi_nodes, f"Portal seed {node_id} must be in ROI"
            assert global_to_roi[node_id] >= 0, f"Portal seed {node_id} must have valid mapping"

    def test_roi_geometric_truncation_preserves_critical_nodes(self, mock_pathfinder_small):
        """Test that ROI truncation preserves src, dst, and portal seeds."""
        pf = mock_pathfinder_small
        pf.max_roi_nodes = 50  # Force truncation
        
        src, dst = 111, 288
        portal_seeds = [(150, 0.5), (250, 0.5)]
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(
            src, dst, corridor_buffer=5, layer_margin=2, portal_seeds=portal_seeds
        )
        
        # Should be truncated
        assert len(roi_nodes) <= pf.max_roi_nodes, \
            f"ROI should be truncated to {pf.max_roi_nodes}, got {len(roi_nodes)}"
        
        # But critical nodes must still be present
        assert src in roi_nodes, "Src must survive truncation"
        assert dst in roi_nodes, "Dst must survive truncation"
        for node_id, _ in portal_seeds:
            assert node_id in roi_nodes, f"Portal seed {node_id} must survive truncation"

    def test_roi_geometric_deterministic(self, mock_pathfinder_small, basic_nodes):
        """Test that geometric ROI extraction is deterministic."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi1, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=3, layer_margin=1)
        roi2, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=3, layer_margin=1)
        
        # Should produce identical results
        assert len(roi1) == len(roi2), "ROI size should be deterministic"
        assert np.array_equal(np.sort(roi1), np.sort(roi2)), \
            "ROI nodes should be identical (order may vary)"

    def test_roi_geometric_bounds_clamping(self, mock_pathfinder_small):
        """Test that ROI bounds are clamped to grid limits."""
        pf = mock_pathfinder_small
        
        # Src near edge (0, 0, 1)
        src = 100  # layer 1, y=0, x=0
        # Dst near opposite edge (9, 9, 2)
        dst = 299  # layer 2, y=9, x=9
        
        # Large buffer that would exceed grid bounds
        roi_nodes, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=20, layer_margin=5)
        
        # Verify all nodes are within grid bounds
        for node in roi_nodes:
            x, y, z = pf.lattice.idx_to_coord(int(node))
            assert 0 <= x < pf.lattice.x_steps, f"X coord {x} out of bounds"
            assert 0 <= y < pf.lattice.y_steps, f"Y coord {y} out of bounds"
            assert 0 <= z < pf.lattice.layers, f"Z coord {z} out of bounds"


# ============================================================================
# TestROIBFS - BFS-based ROI extraction
# ============================================================================

class TestROIBFS:
    """Test extract_roi_bfs() bidirectional BFS approach."""

    def test_roi_bfs_includes_src_and_dst(self, mock_pathfinder_small, basic_nodes):
        """Test that BFS ROI always includes src and dst nodes."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        # Need to provide minimal graph connectivity for BFS
        # Create empty indptr/indices for now
        pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
        pf.graph.indices = np.array([], dtype=np.int32)
        
        roi_nodes, global_to_roi = pf.extract_roi_bfs(src, dst, initial_radius=10)
        
        assert src in roi_nodes, f"Src {src} must be in ROI"
        assert dst in roi_nodes, f"Dst {dst} must be in ROI"
        assert global_to_roi[src] >= 0, "Src must have valid ROI mapping"
        assert global_to_roi[dst] >= 0, "Dst must have valid ROI mapping"

    def test_roi_bfs_respects_radius(self, mock_pathfinder_small):
        """Test that initial_radius parameter controls BFS depth."""
        pf = mock_pathfinder_small
        src, dst = 111, 288
        
        # Setup minimal graph
        pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
        pf.graph.indices = np.array([], dtype=np.int32)
        
        # Small radius
        roi_small, _ = pf.extract_roi_bfs(src, dst, initial_radius=5)
        
        # Large radius
        roi_large, _ = pf.extract_roi_bfs(src, dst, initial_radius=20)
        
        # Larger radius should explore more nodes (unless truncated)
        # With empty graph, both might just return src+dst, so this is a weak assertion
        assert len(roi_large) >= len(roi_small), \
            f"Larger radius should not decrease ROI size: small={len(roi_small)}, large={len(roi_large)}"

    def test_roi_bfs_includes_portal_seeds(self, mock_pathfinder_small):
        """Test that BFS ROI preserves portal seeds."""
        pf = mock_pathfinder_small
        src, dst = 111, 288
        
        pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
        pf.graph.indices = np.array([], dtype=np.int32)
        
        portal_seeds = [(150, 0.5), (250, 0.5)]
        
        roi_nodes, global_to_roi = pf.extract_roi_bfs(
            src, dst, initial_radius=10, portal_seeds=portal_seeds
        )
        
        # Verify all portal seeds are in ROI
        for node_id, _ in portal_seeds:
            assert node_id in roi_nodes, f"Portal seed {node_id} must be in ROI"
            assert global_to_roi[node_id] >= 0, f"Portal seed {node_id} must have valid mapping"

    def test_roi_bfs_truncation_preserves_critical_nodes(self, mock_pathfinder_small):
        """Test that BFS truncation preserves src, dst, and portal seeds."""
        pf = mock_pathfinder_small
        pf.max_roi_nodes = 20  # Force truncation
        
        pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
        pf.graph.indices = np.array([], dtype=np.int32)
        
        src, dst = 111, 288
        portal_seeds = [(150, 0.5), (250, 0.5)]
        
        roi_nodes, _ = pf.extract_roi_bfs(
            src, dst, initial_radius=50, portal_seeds=portal_seeds
        )
        
        # Critical nodes must be present even after truncation
        assert src in roi_nodes, "Src must survive truncation"
        assert dst in roi_nodes, "Dst must survive truncation"
        for node_id, _ in portal_seeds:
            assert node_id in roi_nodes, f"Portal seed {node_id} must survive truncation"

    def test_roi_bfs_deterministic(self, mock_pathfinder_small, basic_nodes):
        """Test that BFS ROI extraction is deterministic."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        pf.graph.indptr = np.zeros(pf.N + 1, dtype=np.int32)
        pf.graph.indices = np.array([], dtype=np.int32)
        
        roi1, _ = pf.extract_roi_bfs(src, dst, initial_radius=10)
        roi2, _ = pf.extract_roi_bfs(src, dst, initial_radius=10)
        
        # Should produce identical results
        assert len(roi1) == len(roi2), "ROI size should be deterministic"
        assert np.array_equal(np.sort(roi1), np.sort(roi2)), \
            "ROI nodes should be identical"


# ============================================================================
# TestROIGlobalToROIMapping - Mapping validation
# ============================================================================

class TestROIGlobalToROIMapping:
    """Test global_to_roi mapping correctness."""

    def test_global_to_roi_maps_roi_nodes(self, mock_pathfinder_small, basic_nodes):
        """Test that global_to_roi correctly maps ROI nodes to local indices."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(src, dst)
        
        # Every ROI node should have a valid mapping
        for local_idx, global_node in enumerate(roi_nodes):
            assert global_to_roi[global_node] == local_idx, \
                f"Node {global_node} should map to local index {local_idx}, got {global_to_roi[global_node]}"

    def test_global_to_roi_marks_non_roi_nodes_negative(self, mock_pathfinder_small, basic_nodes):
        """Test that non-ROI nodes are marked with -1 in mapping."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(src, dst, corridor_buffer=2)
        
        # Find a node definitely outside ROI (far corner)
        far_node = 0  # (0, 0, 0)
        if far_node not in roi_nodes:
            assert global_to_roi[far_node] == -1, \
                f"Non-ROI node {far_node} should be marked -1, got {global_to_roi[far_node]}"

    def test_global_to_roi_size_matches_graph(self, mock_pathfinder_small, basic_nodes):
        """Test that global_to_roi array covers entire graph."""
        pf = mock_pathfinder_small
        src, dst = basic_nodes["src"], basic_nodes["dst"]
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(src, dst)
        
        assert len(global_to_roi) == pf.N, \
            f"global_to_roi should cover all {pf.N} nodes, got {len(global_to_roi)}"


# ============================================================================
# TestROIEdgeCases - Edge case validation
# ============================================================================

class TestROIEdgeCases:
    """Test ROI extraction edge cases."""

    def test_roi_src_equals_dst(self, mock_pathfinder_small):
        """Test ROI extraction when src == dst (degenerate case)."""
        pf = mock_pathfinder_small
        src = dst = 150
        
        roi_nodes, global_to_roi = pf.extract_roi_geometric(src, dst, corridor_buffer=2)
        
        # Should still create valid ROI
        assert src in roi_nodes, "Src/dst node must be in ROI"
        assert len(roi_nodes) > 0, "ROI should not be empty"

    def test_roi_adjacent_nodes(self, mock_pathfinder_small):
        """Test ROI extraction for adjacent src/dst (minimal case)."""
        pf = mock_pathfinder_small
        src = 111  # (1, 1, 1)
        dst = 112  # (2, 1, 1) - adjacent in X direction
        
        roi_nodes, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=1)
        
        assert src in roi_nodes, "Src must be in ROI"
        assert dst in roi_nodes, "Dst must be in ROI"
        assert len(roi_nodes) >= 2, "ROI should include at least src and dst"

    def test_roi_across_full_board(self, mock_pathfinder_small):
        """Test ROI extraction for maximum distance route."""
        pf = mock_pathfinder_small
        src = 100  # (0, 0, 1) - near corner
        dst = 399  # (9, 9, 3) - opposite corner
        
        roi_nodes, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=2)
        
        assert src in roi_nodes, "Src must be in ROI"
        assert dst in roi_nodes, "Dst must be in ROI"
        # Should create substantial corridor
        assert len(roi_nodes) > 100, f"Long route should have large ROI, got {len(roi_nodes)}"

    def test_roi_zero_buffer_still_includes_nodes(self, mock_pathfinder_small):
        """Test that zero buffer still creates valid ROI."""
        pf = mock_pathfinder_small
        src, dst = 111, 288
        
        roi_nodes, _ = pf.extract_roi_geometric(src, dst, corridor_buffer=0, layer_margin=0)
        
        # Even with zero buffer, should have L-corridor nodes
        assert src in roi_nodes, "Src must be in ROI"
        assert dst in roi_nodes, "Dst must be in ROI"
        assert len(roi_nodes) > 2, "Zero buffer should still create corridor"
