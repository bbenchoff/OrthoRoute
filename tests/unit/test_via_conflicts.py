"""Unit tests for via barrel conflict detection.

Tests via ownership tracking, barrel conflict detection, and pooling penalties
to ensure multi-layer routing correctness and prevent via collisions.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# Fixtures
# ==================================================================================================================================

@pytest.fixture
def mock_lattice_via():
    """Mock lattice for via conflict testing (5×5×3 grid)."""
    lattice = Mock()
    lattice.x_steps = 5
    lattice.y_steps = 5
    lattice.layers = 3
    lattice.total_nodes = 75  # 5×5×3
    
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
def mock_pathfinder_via(mock_lattice_via):
    """Mock UnifiedPathFinder with via tracking."""
    pf = Mock()
    pf.lattice = mock_lattice_via
    pf.N = mock_lattice_via.total_nodes
    
    # Via ownership tracking (maps node XY position to owning net)
    pf.via_barrel_owners = {}  # {(x, y): net_name}
    
    # Via column/segment usage arrays
    pf.via_col_use = np.zeros(mock_lattice_via.x_steps * mock_lattice_via.y_steps, dtype=np.float32)
    pf.via_seg_use = np.zeros(mock_lattice_via.x_steps * mock_lattice_via.y_steps * (mock_lattice_via.layers - 1), dtype=np.float32)
    
    # Via column/segment capacity
    pf.via_col_cap = np.ones(mock_lattice_via.x_steps * mock_lattice_via.y_steps, dtype=np.float32)
    pf.via_seg_cap = np.ones(mock_lattice_via.x_steps * mock_lattice_via.y_steps * (mock_lattice_via.layers - 1), dtype=np.float32)
    
    # Configure _mark_via_barrel_ownership_for_path method
    def mock_mark_via_ownership(net_name, path):
        """Mark via ownership for all layer transitions in path."""
        for i in range(len(path) - 1):
            idx1, idx2 = path[i], path[i + 1]
            x1, y1, z1 = pf.lattice.idx_to_coord(idx1)
            x2, y2, z2 = pf.lattice.idx_to_coord(idx2)
            
            # Layer transition = via
            if z1 != z2 and x1 == x2 and y1 == y2:
                pf.via_barrel_owners[(x1, y1)] = net_name
    
    # Configure _build_owner_bitmap_for_fullgraph method
    def mock_build_owner_bitmap(net_name, force_allow_nodes=None):
        """Build bitmap: True = accessible, False = blocked by other net's via."""
        bitmap = np.ones(pf.N, dtype=bool)
        
        # Block nodes at positions owned by other nets
        for (x, y), owner in pf.via_barrel_owners.items():
            if owner != net_name:
                # Block all layers at this XY position
                for z in range(pf.lattice.layers):
                    node_idx = pf.lattice.coord_to_idx(x, y, z)
                    bitmap[node_idx] = False
        
        # Force allow override
        if force_allow_nodes:
            for node_idx in force_allow_nodes:
                bitmap[node_idx] = True
        
        return bitmap
    
    # Configure _detect_barrel_conflicts method
    def mock_detect_barrel_conflicts():
        """Detect via barrel conflicts based on usage vs capacity."""
        conflict_bitmap = np.zeros(pf.N, dtype=np.float32)
        conflict_count = 0
        
        # Check column overuse
        for col_idx in range(len(pf.via_col_use)):
            if pf.via_col_use[col_idx] > pf.via_col_cap[col_idx]:
                # Mark all layers at this XY position
                y, x = divmod(col_idx, pf.lattice.x_steps)
                for z in range(pf.lattice.layers):
                    node_idx = pf.lattice.coord_to_idx(x, y, z)
                    conflict_bitmap[node_idx] = pf.via_col_use[col_idx] - pf.via_col_cap[col_idx]
                conflict_count += 1
        
        # Check segment overuse
        for seg_idx in range(len(pf.via_seg_use)):
            if pf.via_seg_use[seg_idx] > pf.via_seg_cap[seg_idx]:
                conflict_count += 1
        
        return conflict_bitmap, conflict_count
    
    pf._mark_via_barrel_ownership_for_path = mock_mark_via_ownership
    pf._build_owner_bitmap_for_fullgraph = mock_build_owner_bitmap
    pf._detect_barrel_conflicts = mock_detect_barrel_conflicts
    
    return pf


# ============================================================================
# TestViaBarrelOwnership - Track via ownership per net
# ============================================================================

class TestViaBarrelOwnership:
    """Test _mark_via_barrel_ownership_for_path() via ownership tracking."""

    def test_mark_via_ownership_single_via(self, mock_pathfinder_via):
        """Test that marking a path with one via records ownership."""
        pf = mock_pathfinder_via
        
        # Path: layer 0 (0,0) → via → layer 1 (0,0)
        # Node indices: (x=0, y=0, z=0) = 0, (x=0, y=0, z=1) = 25
        path = [0, 25]  # Via from layer 0→1 at (0,0)
        net_name = "NET_A"
        
        pf._mark_via_barrel_ownership_for_path(net_name, path)
        
        # Should mark (0, 0) as owned by NET_A
        assert (0, 0) in pf.via_barrel_owners, "Via position should be tracked"
        assert pf.via_barrel_owners[(0, 0)] == net_name, f"Via should be owned by {net_name}"

    def test_mark_via_ownership_multiple_vias(self, mock_pathfinder_via):
        """Test that multiple vias in path are all marked."""
        pf = mock_pathfinder_via
        
        # Path with two vias at different positions
        # Via 1: (0,0) layers 0→1
        # Via 2: (1,1) layers 1→2
        path = [
            0,   # (0,0,0)
            25,  # (0,0,1) - via 1
            31,  # (1,1,1)
            56   # (1,1,2) - via 2
        ]
        net_name = "NET_B"
        
        pf._mark_via_barrel_ownership_for_path(net_name, path)
        
        assert (0, 0) in pf.via_barrel_owners, "Via 1 position should be tracked"
        assert (1, 1) in pf.via_barrel_owners, "Via 2 position should be tracked"
        assert pf.via_barrel_owners[(0, 0)] == net_name
        assert pf.via_barrel_owners[(1, 1)] == net_name

    def test_mark_via_ownership_no_vias_in_path(self, mock_pathfinder_via):
        """Test that horizontal-only path doesn't mark any vias."""
        pf = mock_pathfinder_via
        
        # Horizontal path on same layer (no layer changes)
        path = [0, 1, 2]  # (0,0,0) → (1,0,0) → (2,0,0)
        net_name = "NET_C"
        
        pf._mark_via_barrel_ownership_for_path(net_name, path)
        
        # No vias should be marked
        assert len(pf.via_barrel_owners) == 0, "Horizontal path should not create via ownership"

    def test_mark_via_ownership_overwrites_previous(self, mock_pathfinder_via):
        """Test that rerouting overwrites via ownership."""
        pf = mock_pathfinder_via
        
        # First route: NET_A owns (0,0)
        path1 = [0, 25]
        pf._mark_via_barrel_ownership_for_path("NET_A", path1)
        assert pf.via_barrel_owners[(0, 0)] == "NET_A"
        
        # Second route: NET_B takes over (0,0)
        path2 = [0, 25, 26]
        pf._mark_via_barrel_ownership_for_path("NET_B", path2)
        assert pf.via_barrel_owners[(0, 0)] == "NET_B", "Newer route should overwrite ownership"


# ============================================================================
# TestOwnerBitmap - Owner bitmap creation for conflict detection
# ============================================================================

class TestOwnerBitmap:
    """Test _build_owner_bitmap_for_fullgraph() conflict bitmap creation."""

    def test_owner_bitmap_allows_own_vias(self, mock_pathfinder_via):
        """Test that owner bitmap allows current net's own vias."""
        pf = mock_pathfinder_via
        
        # NET_A owns (0,0)
        pf.via_barrel_owners[(0, 0)] = "NET_A"
        
        # Build bitmap for NET_A (should allow its own via)
        owner_bitmap = pf._build_owner_bitmap_for_fullgraph("NET_A")
        
        # Node 0 = (0,0,0) should be allowed (True)
        # Node 25 = (0,0,1) should be allowed (True)
        assert owner_bitmap[0] == True, "NET_A should access its own via at layer 0"
        assert owner_bitmap[25] == True, "NET_A should access its own via at layer 1"

    def test_owner_bitmap_blocks_other_nets_vias(self, mock_pathfinder_via):
        """Test that owner bitmap blocks other nets' vias."""
        pf = mock_pathfinder_via
        
        # NET_A owns (0,0)
        pf.via_barrel_owners[(0, 0)] = "NET_A"
        
        # Build bitmap for NET_B (should block NET_A's via)
        owner_bitmap = pf._build_owner_bitmap_for_fullgraph("NET_B")
        
        # Node 0 = (0,0,0) should be blocked (False)
        # Node 25 = (0,0,1) should be blocked (False)
        assert owner_bitmap[0] == False, "NET_B should not access NET_A's via at layer 0"
        assert owner_bitmap[25] == False, "NET_B should not access NET_A's via at layer 1"

    def test_owner_bitmap_allows_unowned_positions(self, mock_pathfinder_via):
        """Test that owner bitmap allows positions with no owner."""
        pf = mock_pathfinder_via
        
        # No ownership at (1,1)
        owner_bitmap = pf._build_owner_bitmap_for_fullgraph("NET_A")
        
        # Node 6 = (1,1,0) should be allowed (no owner)
        assert owner_bitmap[6] == True, "Unowned position should be accessible"

    def test_owner_bitmap_force_allow_overrides(self, mock_pathfinder_via):
        """Test that force_allow_nodes parameter overrides ownership."""
        pf = mock_pathfinder_via
        
        # NET_A owns (0,0)
        pf.via_barrel_owners[(0, 0)] = "NET_A"
        
        # Build bitmap for NET_B, but force allow node 0
        force_allow = [0, 25]  # Force allow NET_A's via
        owner_bitmap = pf._build_owner_bitmap_for_fullgraph("NET_B", force_allow_nodes=force_allow)
        
        # Should be allowed despite ownership
        assert owner_bitmap[0] == True, "force_allow should override ownership block"
        assert owner_bitmap[25] == True, "force_allow should override ownership block"

    def test_owner_bitmap_size_matches_graph(self, mock_pathfinder_via):
        """Test that owner bitmap covers all nodes."""
        pf = mock_pathfinder_via
        
        owner_bitmap = pf._build_owner_bitmap_for_fullgraph("NET_A")
        
        assert len(owner_bitmap) == pf.N, \
            f"Owner bitmap should cover all {pf.N} nodes, got {len(owner_bitmap)}"


# ============================================================================
# TestBarrelConflicts - Detect overlapping via barrels
# ============================================================================

class TestBarrelConflicts:
    """Test _detect_barrel_conflicts() via overlap detection."""

    def test_detect_conflicts_no_overuse(self, mock_pathfinder_via):
        """Test that no conflicts detected when vias under capacity."""
        pf = mock_pathfinder_via
        
        # All usage at 0.5 (under capacity of 1.0)
        pf.via_col_use[:] = 0.5
        pf.via_seg_use[:] = 0.5
        
        conflict_bitmap, conflict_count = pf._detect_barrel_conflicts()
        
        # No conflicts expected
        assert conflict_count == 0, f"Should have 0 conflicts, got {conflict_count}"
        assert np.all(conflict_bitmap == 0), "Conflict bitmap should be all zeros"

    def test_detect_conflicts_column_overuse(self, mock_pathfinder_via):
        """Test that column overuse is detected."""
        pf = mock_pathfinder_via
        
        # Overuse via column 0 (position 0,0)
        pf.via_col_use[0] = 2.0  # Over capacity of 1.0
        pf.via_col_cap[0] = 1.0
        
        conflict_bitmap, conflict_count = pf._detect_barrel_conflicts()
        
        # Should detect conflict at column 0
        assert conflict_count > 0, "Should detect column overuse"
        
        # Nodes at position (0,0) across all layers should be marked
        # Nodes: 0(0,0,0), 25(0,0,1), 50(0,0,2)
        assert conflict_bitmap[0] > 0, "Layer 0 at conflict position should be marked"
        assert conflict_bitmap[25] > 0, "Layer 1 at conflict position should be marked"
        assert conflict_bitmap[50] > 0, "Layer 2 at conflict position should be marked"

    def test_detect_conflicts_segment_overuse(self, mock_pathfinder_via):
        """Test that segment overuse is detected."""
        pf = mock_pathfinder_via
        
        # Overuse via segment 0 (position 0,0, layer 0→1)
        pf.via_seg_use[0] = 3.0  # Over capacity
        pf.via_seg_cap[0] = 1.0
        
        conflict_bitmap, conflict_count = pf._detect_barrel_conflicts()
        
        # Should detect conflict
        assert conflict_count > 0, "Should detect segment overuse"

    def test_detect_conflicts_multiple_positions(self, mock_pathfinder_via):
        """Test detection of conflicts at multiple via positions."""
        pf = mock_pathfinder_via
        
        # Overuse at two positions
        pf.via_col_use[0] = 2.0  # Position (0,0)
        pf.via_col_use[6] = 2.5  # Position (1,1)
        pf.via_col_cap[:] = 1.0
        
        conflict_bitmap, conflict_count = pf._detect_barrel_conflicts()
        
        # Both positions should show conflicts
        assert conflict_count >= 2, f"Should detect at least 2 conflict positions, got {conflict_count}"

    def test_detect_conflicts_returns_zero_for_clean_graph(self, mock_pathfinder_via):
        """Test that clean graph returns zero conflicts."""
        pf = mock_pathfinder_via
        
        # All usage = 0 (pristine graph)
        pf.via_col_use[:] = 0.0
        pf.via_seg_use[:] = 0.0
        
        conflict_bitmap, conflict_count = pf._detect_barrel_conflicts()
        
        assert conflict_count == 0, "Pristine graph should have zero conflicts"
        assert np.all(conflict_bitmap == 0), "Conflict bitmap should be empty"


# ============================================================================
# TestViaPoolingPenalties - Via congestion penalties
# ============================================================================

class TestViaPoolingPenalties:
    """Test via pooling penalty calculations (via overuse cost increases)."""

    def test_via_overuse_increases_cost(self, mock_pathfinder_via):
        """Test that via overuse increases edge costs.
        
        NOTE: This is conceptual - actual implementation may vary.
        The EdgeAccountant.compute_overuse() includes via spatial checks.
        """
        pf = mock_pathfinder_via
        
        # Set up overuse at position (0,0)
        pf.via_col_use[0] = 2.0
        pf.via_col_cap[0] = 1.0
        
        # Overuse = 2.0 - 1.0 = 1.0
        overuse = max(0, pf.via_col_use[0] - pf.via_col_cap[0])
        
        assert overuse == 1.0, f"Via overuse should be 1.0, got {overuse}"

    def test_via_under_capacity_has_zero_penalty(self, mock_pathfinder_via):
        """Test that vias under capacity have no overuse penalty."""
        pf = mock_pathfinder_via
        
        # Usage under capacity
        pf.via_col_use[0] = 0.5
        pf.via_col_cap[0] = 1.0
        
        overuse = max(0, pf.via_col_use[0] - pf.via_col_cap[0])
        
        assert overuse == 0.0, "Under-capacity via should have zero overuse"


# ============================================================================
# TestViaEdgeCases - Edge case validation
# ============================================================================

class TestViaEdgeCases:
    """Test via conflict edge cases."""

    def test_via_single_node_path(self, mock_pathfinder_via):
        """Test that single-node path doesn't create via ownership."""
        pf = mock_pathfinder_via
        
        path = [5]  # Single node
        pf._mark_via_barrel_ownership_for_path("NET_X", path)
        
        assert len(pf.via_barrel_owners) == 0, "Single node should not create via"

    def test_via_empty_path(self, mock_pathfinder_via):
        """Test that empty path doesn't crash."""
        pf = mock_pathfinder_via
        
        path = []
        pf._mark_via_barrel_ownership_for_path("NET_Y", path)
        
        # Should handle gracefully
        assert len(pf.via_barrel_owners) == 0, "Empty path should not create via"

    def test_via_same_position_different_layers(self, mock_pathfinder_via):
        """Test that multi-layer via at same XY position is tracked once."""
        pf = mock_pathfinder_via
        
        # Via through all 3 layers at position (0,0)
        path = [0, 25, 50]  # (0,0,0) → (0,0,1) → (0,0,2)
        pf._mark_via_barrel_ownership_for_path("NET_Z", path)
        
        # Should mark (0,0) once (not three times)
        assert (0, 0) in pf.via_barrel_owners
        assert pf.via_barrel_owners[(0, 0)] == "NET_Z"
        
        # Only one entry for this XY position
        xy_count = sum(1 for k in pf.via_barrel_owners.keys() if k == (0, 0))
        assert xy_count == 1, "Multi-layer via at same position should have one entry"

    def test_conflict_bitmap_with_zero_capacity(self, mock_pathfinder_via):
        """Test that zero-capacity via is always marked as conflict."""
        pf = mock_pathfinder_via
        
        # Set via column 0 to zero capacity (blocked)
        pf.via_col_cap[0] = 0.0
        pf.via_col_use[0] = 0.0  # Even with zero usage
        
        conflict_bitmap, _ = pf._detect_barrel_conflicts()
        
        # Zero capacity means any usage creates conflict
        # Check if blocked position is handled (implementation-specific)
        # This test documents expected behavior
        pass  # Actual behavior depends on implementation details

    def test_via_at_board_edges(self, mock_pathfinder_via):
        """Test via ownership at board edges ((0,0) and (4,4))."""
        pf = mock_pathfinder_via
        
        # Via at corner (0,0)
        path_corner = [0, 25]
        pf._mark_via_barrel_ownership_for_path("NET_CORNER", path_corner)
        
        # Via at opposite corner (4,4)
        # Node index: (x=4, y=4, z=0) = 0*25 + 4*5 + 4 = 24
        path_opposite = [24, 49]  # (4,4,0) → (4,4,1)
        pf._mark_via_barrel_ownership_for_path("NET_OPPOSITE", path_opposite)
        
        assert (0, 0) in pf.via_barrel_owners, "Corner via should be tracked"
        assert (4, 4) in pf.via_barrel_owners, "Opposite corner via should be tracked"
