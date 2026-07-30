"""Unit tests for real_global_grid.py - Grid indexing and neighbor generation.

Tests grid coordinate system, bounds checking, and neighbor determinism
to prevent OOB crashes and ensure reproducible routing.
"""
import pytest
import numpy as np
from orthoroute.algorithms.manhattan.real_global_grid import (
    GridShape,
    gid,
    xyz_from_gid,
    neighbors_for_gid,
    validate_path_bounds,
    validate_edges_from_path,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_grid():
    """4×4×2 grid for basic testing."""
    return GridShape(NL=2, NX=4, NY=4)


@pytest.fixture
def large_grid():
    """106×234×18 grid (TestBackplane dimensions)."""
    return GridShape(NL=18, NX=106, NY=234)


@pytest.fixture
def single_layer_grid():
    """Single-layer 3×3 grid for edge case testing."""
    return GridShape(NL=1, NX=3, NY=3)


@pytest.fixture
def allowed_transitions_manhattan():
    """Manhattan routing: only ±1 layer transitions."""
    # Layer 0 can go to layer 1, layer 1 can go to 0 or 2, etc.
    return {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2],
    }


# ============================================================================
# TestGridShape - Immutable dimension container
# ============================================================================

class TestGridShape:
    """Test GridShape dataclass creation, validation, and immutability."""

    def test_gridshape_creation_valid_dimensions(self, small_grid):
        """Test that GridShape constructs with positive dimensions."""
        assert small_grid.NL == 2
        assert small_grid.NX == 4
        assert small_grid.NY == 4

    def test_gridshape_total_nodes_calculation(self, small_grid):
        """Test that total_nodes property computes correctly (NL × NX × NY)."""
        expected = 2 * 4 * 4
        assert small_grid.total_nodes == expected

    def test_gridshape_xy_property(self, small_grid):
        """Test that XY property computes correctly (NX × NY)."""
        expected = 4 * 4
        assert small_grid.XY == expected

    def test_gridshape_immutable(self, small_grid):
        """Test that GridShape is frozen (dataclass mutation rejected).
        
        This prevents accidental graph dimension changes during routing.
        """
        with pytest.raises((AttributeError, TypeError)):
            small_grid.NL = 99  # frozen dataclass must raise

    def test_gridshape_rejects_zero_dimensions(self):
        """Test that GridShape raises AssertionError for zero dimensions.
        
        Prevents degenerate graphs that would cause division by zero.
        """
        with pytest.raises(AssertionError):
            GridShape(NL=0, NX=4, NY=4)

    def test_gridshape_rejects_negative_dimensions(self):
        """Test that GridShape raises AssertionError for negative dimensions.
        
        Prevents invalid graph construction.
        """
        with pytest.raises(AssertionError):
            GridShape(NL=2, NX=-4, NY=4)


# ============================================================================
# TestGidConversion - Global ID coordinate conversion
# ============================================================================

class TestGidConversion:
    """Test gid() conversion from (layer, x, y) to global ID."""

    def test_gid_converts_origin(self, small_grid):
        """Test that gid() converts (0,0,0) to 0 (origin node)."""
        result = gid(small_grid, layer=0, x=0, y=0)
        assert result == 0

    def test_gid_converts_valid_coordinates(self, small_grid):
        """Test that gid() returns correct 1D index for valid coordinates.
        
        Formula: gid = layer * (NX * NY) + y * NX + x
        For (layer=1, x=2, y=1) in 4×4×2 grid:
        gid = 1 * 16 + 1 * 4 + 2 = 22
        """
        result = gid(small_grid, layer=1, x=2, y=1)
        expected = 1 * 16 + 1 * 4 + 2
        assert result == expected

    def test_gid_rejects_negative_layer(self, small_grid):
        """Test that gid() raises AssertionError for negative layer indices.
        
        This prevents OOB access to the lattice node array that would cause
        silent data corruption or segfaults in the CUDA kernel.
        """
        with pytest.raises(AssertionError, match="Layer .* OOB"):
            gid(small_grid, layer=-1, x=0, y=0)

    def test_gid_rejects_layer_out_of_bounds(self, small_grid):
        """Test that gid() raises AssertionError when layer >= NL.
        
        Prevents accessing beyond allocated layer array.
        """
        with pytest.raises(AssertionError, match="Layer .* OOB"):
            gid(small_grid, layer=2, x=0, y=0)  # NL=2, so max layer is 1

    def test_gid_rejects_negative_x(self, small_grid):
        """Test that gid() raises AssertionError for negative x indices."""
        with pytest.raises(AssertionError, match="X .* OOB"):
            gid(small_grid, layer=0, x=-1, y=0)

    def test_gid_rejects_x_out_of_bounds(self, small_grid):
        """Test that gid() raises AssertionError when x >= NX."""
        with pytest.raises(AssertionError, match="X .* OOB"):
            gid(small_grid, layer=0, x=4, y=0)  # NX=4, so max x is 3

    def test_gid_rejects_negative_y(self, small_grid):
        """Test that gid() raises AssertionError for negative y indices."""
        with pytest.raises(AssertionError, match="Y .* OOB"):
            gid(small_grid, layer=0, x=0, y=-1)

    def test_gid_rejects_y_out_of_bounds(self, small_grid):
        """Test that gid() raises AssertionError when y >= NY."""
        with pytest.raises(AssertionError, match="Y .* OOB"):
            gid(small_grid, layer=0, x=0, y=4)  # NY=4, so max y is 3


# ============================================================================
# TestXyzFromGid - Reverse conversion
# ============================================================================

class TestXyzFromGid:
    """Test xyz_from_gid() conversion from global ID to (layer, x, y)."""

    def test_xyz_from_gid_converts_origin(self, small_grid):
        """Test that xyz_from_gid(0) returns (0, 0, 0) (origin node)."""
        layer, x, y = xyz_from_gid(small_grid, 0)
        assert (layer, x, y) == (0, 0, 0)

    def test_xyz_from_gid_round_trip(self, small_grid):
        """Test that xyz_from_gid(gid(...)) == (layer, x, y) (identity).
        
        This validates bidirectional conversion correctness.
        """
        original = (1, 2, 1)
        g = gid(small_grid, *original)
        recovered = xyz_from_gid(small_grid, g)
        assert recovered == original

    def test_xyz_from_gid_all_nodes_round_trip(self, single_layer_grid):
        """Test round-trip conversion for all nodes in a small grid.
        
        This ensures no indexing bugs in corner cases.
        """
        for layer in range(single_layer_grid.NL):
            for y in range(single_layer_grid.NY):
                for x in range(single_layer_grid.NX):
                    g = gid(single_layer_grid, layer, x, y)
                    recovered = xyz_from_gid(single_layer_grid, g)
                    assert recovered == (layer, x, y), \
                        f"Round-trip failed for ({layer}, {x}, {y})"

    def test_xyz_from_gid_rejects_negative_gid(self, small_grid):
        """Test that xyz_from_gid() raises AssertionError for negative GID."""
        with pytest.raises(AssertionError, match="GID .* OOB"):
            xyz_from_gid(small_grid, -1)

    def test_xyz_from_gid_rejects_gid_out_of_bounds(self, small_grid):
        """Test that xyz_from_gid() raises AssertionError when GID >= total_nodes."""
        total = small_grid.total_nodes
        with pytest.raises(AssertionError, match="GID .* OOB"):
            xyz_from_gid(small_grid, total)  # max valid GID is total-1


# ============================================================================
# TestNeighborGeneration - Deterministic neighbor order
# ============================================================================

class TestNeighborGeneration:
    """Test neighbors_for_gid() determinism and bounds checking."""

    def test_neighbors_deterministic_order(self, small_grid, allowed_transitions_manhattan):
        """Test that neighbors are returned in order: E, W, N, S, then vias ascending.
        
        This ensures deterministic routing — same board → same route.
        """
        # Node at (layer=0, x=1, y=1) — interior node with all directions available
        g = gid(small_grid, layer=0, x=1, y=1)
        neighbors = neighbors_for_gid(small_grid, g, allowed_transitions_manhattan)
        
        # Extract neighbor coordinates
        coords = [xyz_from_gid(small_grid, ng) for ng, _, _ in neighbors]
        
        # First 4 should be track moves: E, W, N, S
        assert coords[0] == (0, 2, 1), "First neighbor should be East (x+1)"
        assert coords[1] == (0, 0, 1), "Second neighbor should be West (x-1)"
        assert coords[2] == (0, 1, 2), "Third neighbor should be North (y+1)"
        assert coords[3] == (0, 1, 0), "Fourth neighbor should be South (y-1)"
        
        # Fifth should be via to layer 1 (ascending)
        assert coords[4] == (1, 1, 1), "Via neighbor should be to layer 1"

    def test_neighbors_respects_x_bounds(self, small_grid, allowed_transitions_manhattan):
        """Test that neighbors on x-edge don't include OOB indices."""
        # Node at x=0 (left edge) — no West neighbor
        g = gid(small_grid, layer=0, x=0, y=1)
        neighbors = neighbors_for_gid(small_grid, g, allowed_transitions_manhattan)
        
        coords = [xyz_from_gid(small_grid, ng) for ng, _, _ in neighbors]
        
        # Should not have any x=-1 neighbors
        assert all(c[1] >= 0 for c in coords), "All x coordinates must be >= 0"
        assert all(c[1] < small_grid.NX for c in coords), "All x coordinates must be < NX"

    def test_neighbors_respects_y_bounds(self, small_grid, allowed_transitions_manhattan):
        """Test that neighbors on y-edge don't include OOB indices."""
        # Node at y=0 (bottom edge) — no South neighbor
        g = gid(small_grid, layer=0, x=1, y=0)
        neighbors = neighbors_for_gid(small_grid, g, allowed_transitions_manhattan)
        
        coords = [xyz_from_gid(small_grid, ng) for ng, _, _ in neighbors]
        
        # Should not have any y=-1 neighbors
        assert all(c[2] >= 0 for c in coords), "All y coordinates must be >= 0"
        assert all(c[2] < small_grid.NY for c in coords), "All y coordinates must be < NY"

    def test_neighbors_via_pairs_legal(self, allowed_transitions_manhattan):
        """Test that vias only span legal layer pairs from allowed_transitions.
        
        This enforces routing rules (e.g., Manhattan ±1 only).
        """
        grid = GridShape(NL=4, NX=3, NY=3)
        
        # Node at layer 1 — can via to layers 0 and 2
        g = gid(grid, layer=1, x=1, y=1)
        neighbors = neighbors_for_gid(grid, g, allowed_transitions_manhattan)
        
        # Extract via neighbors (is_via=True)
        via_neighbors = [(ng, cost, is_via) for ng, cost, is_via in neighbors if is_via]
        via_layers = [xyz_from_gid(grid, ng)[0] for ng, _, _ in via_neighbors]
        
        # Should only have vias to layers 0 and 2 (from allowed_transitions[1])
        assert sorted(via_layers) == [0, 2], \
            f"Layer 1 should only via to [0, 2], got {via_layers}"

    def test_neighbors_exclude_vias_when_disabled(self, small_grid, allowed_transitions_manhattan):
        """Test that include_vias=False excludes via neighbors."""
        g = gid(small_grid, layer=0, x=1, y=1)
        neighbors = neighbors_for_gid(small_grid, g, allowed_transitions_manhattan, 
                                     include_vias=False)
        
        # Should only have 4 track neighbors (E, W, N, S)
        assert len(neighbors) == 4, f"Expected 4 track neighbors, got {len(neighbors)}"
        assert all(not is_via for _, _, is_via in neighbors), \
            "All neighbors should be tracks (is_via=False)"

    def test_neighbors_corner_node(self, small_grid, allowed_transitions_manhattan):
        """Test neighbor count for corner node (fewest neighbors)."""
        # Corner at (0, 0, 0) — only E, N, and via to layer 1
        g = gid(small_grid, layer=0, x=0, y=0)
        neighbors = neighbors_for_gid(small_grid, g, allowed_transitions_manhattan)
        
        # Should have 2 track neighbors + 1 via = 3 total
        track_count = sum(1 for _, _, is_via in neighbors if not is_via)
        via_count = sum(1 for _, _, is_via in neighbors if is_via)
        
        assert track_count == 2, f"Corner should have 2 track neighbors, got {track_count}"
        assert via_count == 1, f"Corner should have 1 via neighbor, got {via_count}"


# ============================================================================
# TestPathValidation - Bounds and edge validation
# ============================================================================

class TestPathValidation:
    """Test validate_path_bounds() and validate_edges_from_path()."""

    def test_validate_path_bounds_valid_path(self, small_grid):
        """Test that validate_path_bounds() accepts valid path."""
        path = np.array([0, 1, 2, 3], dtype=np.int32)
        assert validate_path_bounds(small_grid, path, "test_net")

    def test_validate_path_bounds_rejects_oob_max(self, small_grid):
        """Test that validate_path_bounds() rejects path with GID >= total_nodes."""
        path = np.array([0, 1, 999], dtype=np.int32)  # 999 > total_nodes
        assert not validate_path_bounds(small_grid, path, "test_net")

    def test_validate_path_bounds_rejects_negative(self, small_grid):
        """Test that validate_path_bounds() rejects path with negative GID."""
        path = np.array([0, -1, 2], dtype=np.int32)
        assert not validate_path_bounds(small_grid, path, "test_net")

    def test_validate_path_bounds_empty_path(self, small_grid):
        """Test that validate_path_bounds() rejects empty path."""
        path = np.array([], dtype=np.int32)
        assert not validate_path_bounds(small_grid, path, "test_net")

    def test_validate_edges_from_path_creates_edges(self):
        """Test that validate_edges_from_path() creates correct edge array."""
        path = np.array([0, 1, 5, 6], dtype=np.int32)
        edges = validate_edges_from_path(path, "test_net")
        
        assert edges.shape == (3, 2), f"Expected (3, 2) edges, got {edges.shape}"
        assert np.array_equal(edges[0], [0, 1])
        assert np.array_equal(edges[1], [1, 5])
        assert np.array_equal(edges[2], [5, 6])

    def test_validate_edges_from_path_rejects_too_short(self):
        """Test that validate_edges_from_path() rejects path with <2 nodes."""
        path = np.array([0], dtype=np.int32)
        with pytest.raises(ValueError, match="Path too short"):
            validate_edges_from_path(path, "test_net")

    def test_validate_edges_from_path_rejects_zero_length(self):
        """Test that validate_edges_from_path() rejects path with duplicate consecutive nodes."""
        path = np.array([0, 1, 1, 2], dtype=np.int32)  # node 1 repeated
        with pytest.raises(ValueError, match="Zero-length edges"):
            validate_edges_from_path(path, "test_net")


# ============================================================================
# Edge Cases and Large Grid Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and large grid behavior."""

    def test_single_cell_grid(self):
        """Test that 1×1×1 grid works correctly."""
        grid = GridShape(NL=1, NX=1, NY=1)
        assert grid.total_nodes == 1
        
        g = gid(grid, layer=0, x=0, y=0)
        assert g == 0
        
        layer, x, y = xyz_from_gid(grid, 0)
        assert (layer, x, y) == (0, 0, 0)

    def test_large_grid_dimensions(self, large_grid):
        """Test that large grid (TestBackplane size) constructs correctly."""
        assert large_grid.NL == 18
        assert large_grid.NX == 106
        assert large_grid.NY == 234
        expected_total = 18 * 106 * 234
        assert large_grid.total_nodes == expected_total

    def test_large_grid_max_coordinate(self, large_grid):
        """Test that max valid coordinate in large grid works."""
        # Max coordinate: (NL-1, NX-1, NY-1)
        g = gid(large_grid, layer=17, x=105, y=233)
        assert g == large_grid.total_nodes - 1
        
        recovered = xyz_from_gid(large_grid, g)
        assert recovered == (17, 105, 233)

    def test_32_layer_grid(self):
        """Test that 32-layer board (max for OrthoRoute) works."""
        grid = GridShape(NL=32, NX=10, NY=10)
        assert grid.NL == 32
        
        # Test max layer
        g = gid(grid, layer=31, x=0, y=0)
        layer, x, y = xyz_from_gid(grid, g)
        assert layer == 31
