"""
NumPy-based implementations of CUDA kernels.

These provide cross-platform compatibility for the pathfinding algorithms
when CUDA is not available. While slower than GPU implementations, they
enable the router to run on any platform including Apple Silicon.
"""
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class NumpyDijkstraKernels:
    """
    NumPy implementations of Dijkstra CUDA kernels.

    These are pure Python/NumPy implementations that provide the same
    functionality as the CUDA kernels but run on CPU.
    """

    def __init__(self, lattice_dims: Optional[Tuple[int, int, int]] = None):
        """
        Initialize NumPy kernels.

        Args:
            lattice_dims: (Nx, Ny, Nz) dimensions of the routing lattice
        """
        self.Nx = lattice_dims[0] if lattice_dims else 0
        self.Ny = lattice_dims[1] if lattice_dims else 0
        self.Nz = lattice_dims[2] if lattice_dims else 0
        self.plane_size = self.Nx * self.Ny

    def set_lattice_dims(self, Nx: int, Ny: int, Nz: int):
        """Set lattice dimensions."""
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.plane_size = Nx * Ny

    def node_to_coords(self, node: int) -> Tuple[int, int, int]:
        """Convert flat node index to (x, y, z) coordinates."""
        z = node // self.plane_size
        remainder = node % self.plane_size
        y = remainder // self.Nx
        x = remainder % self.Nx
        return x, y, z

    def coords_to_node(self, x: int, y: int, z: int) -> int:
        """Convert (x, y, z) coordinates to flat node index."""
        return z * self.plane_size + y * self.Nx + x

    def wavefront_expand(
        self,
        frontier: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        total_cost: np.ndarray,
        dist: np.ndarray,
        parent: np.ndarray,
        roi_bitmap: Optional[np.ndarray] = None,
        use_bitmap: bool = True,
        goal_coords: Optional[Tuple[int, int, int]] = None,
        use_astar: bool = False,
    ) -> np.ndarray:
        """
        Single-source wavefront expansion (one iteration of Dijkstra).

        This is the NumPy equivalent of the wavefront_expand_all CUDA kernel.

        Args:
            frontier: Boolean or bit-packed array of frontier nodes
            indptr: CSR format indptr array
            indices: CSR format indices array
            total_cost: Edge costs (after congestion penalties)
            dist: Distance array (modified in-place)
            parent: Parent array (modified in-place)
            roi_bitmap: Optional bitmap of valid ROI nodes
            use_bitmap: Whether to enforce ROI bitmap
            goal_coords: Optional (x, y, z) goal for A* heuristic
            use_astar: Whether to use A* heuristic

        Returns:
            new_frontier: Boolean array of new frontier nodes
        """
        max_nodes = len(dist)
        new_frontier = np.zeros(max_nodes, dtype=bool)

        # Get frontier node indices
        if frontier.dtype == bool:
            frontier_nodes = np.where(frontier)[0]
        else:
            # Bit-packed frontier - unpack
            frontier_nodes = []
            for word_idx in range(len(frontier)):
                word = frontier[word_idx]
                for bit in range(32):
                    if word & (1 << bit):
                        node = word_idx * 32 + bit
                        if node < max_nodes:
                            frontier_nodes.append(node)
            frontier_nodes = np.array(frontier_nodes, dtype=np.int32)

        if len(frontier_nodes) == 0:
            return new_frontier

        # Process each frontier node
        for node in frontier_nodes:
            node_dist = dist[node]

            # Skip unreachable nodes
            if np.isinf(node_dist):
                continue

            # Get edges for this node
            edge_start = indptr[node]
            edge_end = indptr[node + 1]

            for edge_idx in range(edge_start, edge_end):
                neighbor = indices[edge_idx]

                # Bounds check
                if neighbor < 0 or neighbor >= max_nodes:
                    continue

                # ROI bitmap check
                if use_bitmap and roi_bitmap is not None:
                    word_idx = neighbor >> 5
                    bit_idx = neighbor & 31
                    if word_idx < len(roi_bitmap):
                        if not (roi_bitmap[word_idx] >> bit_idx) & 1:
                            continue
                    else:
                        continue

                edge_cost = total_cost[edge_idx]
                g_new = node_dist + edge_cost

                # A* heuristic
                f_new = g_new
                if use_astar and goal_coords is not None:
                    nx, ny, nz = self.node_to_coords(neighbor)
                    gx, gy, gz = goal_coords
                    h = (abs(gx - nx) + abs(gy - ny)) * 0.4 + abs(gz - nz) * 1.5
                    f_new = g_new + h

                # Relaxation with epsilon guard
                if g_new + 1e-8 < dist[neighbor]:
                    dist[neighbor] = g_new
                    parent[neighbor] = node
                    new_frontier[neighbor] = True

        return new_frontier

    def wavefront_expand_batch(
        self,
        K: int,
        max_roi_size: int,
        frontiers: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        total_cost: np.ndarray,
        dist: np.ndarray,
        parent: np.ndarray,
        roi_bitmaps: Optional[np.ndarray] = None,
        use_bitmap: bool = True,
        goal_coords: Optional[np.ndarray] = None,
        use_astar: bool = False,
        indptr_stride: int = 0,
        indices_stride: int = 0,
        cost_stride: int = 0,
    ) -> np.ndarray:
        """
        Batch wavefront expansion for multiple ROIs.

        Args:
            K: Number of ROIs
            max_roi_size: Maximum nodes per ROI
            frontiers: (K, frontier_words) bit-packed frontiers
            indptr: CSR indptr (shared or per-ROI)
            indices: CSR indices (shared or per-ROI)
            total_cost: Edge costs
            dist: (K, max_roi_size) distance arrays
            parent: (K, max_roi_size) parent arrays
            roi_bitmaps: Optional (K, bitmap_words) ROI bitmaps
            use_bitmap: Whether to enforce bitmaps
            goal_coords: Optional (K, 3) goal coordinates
            use_astar: Whether to use A*
            indptr_stride: Stride for indptr (0 = shared)
            indices_stride: Stride for indices (0 = shared)
            cost_stride: Stride for cost (0 = shared)

        Returns:
            new_frontiers: (K, frontier_words) bit-packed new frontiers
        """
        frontier_words = frontiers.shape[1] if len(frontiers.shape) > 1 else (max_roi_size + 31) // 32
        new_frontiers = np.zeros((K, frontier_words), dtype=np.uint32)

        for roi_idx in range(K):
            # Get ROI-specific arrays
            roi_frontier = frontiers[roi_idx] if len(frontiers.shape) > 1 else frontiers
            roi_dist = dist[roi_idx] if len(dist.shape) > 1 else dist
            roi_parent = parent[roi_idx] if len(parent.shape) > 1 else parent

            # Handle shared vs per-ROI CSR
            if indptr_stride == 0:
                roi_indptr = indptr
                roi_indices = indices
                roi_cost = total_cost
            else:
                roi_indptr = indptr[roi_idx * indptr_stride:(roi_idx + 1) * indptr_stride]
                roi_indices = indices[roi_idx * indices_stride:(roi_idx + 1) * indices_stride]
                roi_cost = total_cost[roi_idx * cost_stride:(roi_idx + 1) * cost_stride]

            # ROI bitmap
            roi_bitmap = None
            if roi_bitmaps is not None:
                bitmap_words = roi_bitmaps.shape[1] if len(roi_bitmaps.shape) > 1 else len(roi_bitmaps)
                roi_bitmap = roi_bitmaps[roi_idx] if len(roi_bitmaps.shape) > 1 else roi_bitmaps

            # Goal coordinates
            goal = None
            if goal_coords is not None and use_astar:
                goal = tuple(goal_coords[roi_idx])

            # Run single ROI expansion
            new_frontier_bool = self.wavefront_expand(
                frontier=roi_frontier,
                indptr=roi_indptr,
                indices=roi_indices,
                total_cost=roi_cost,
                dist=roi_dist,
                parent=roi_parent,
                roi_bitmap=roi_bitmap,
                use_bitmap=use_bitmap,
                goal_coords=goal,
                use_astar=use_astar,
            )

            # Pack new frontier into bit array
            for node in np.where(new_frontier_bool)[0]:
                word_idx = node >> 5
                bit_idx = node & 31
                if word_idx < frontier_words:
                    new_frontiers[roi_idx, word_idx] |= (1 << bit_idx)

        return new_frontiers

    def backtrace_path(
        self,
        parent: np.ndarray,
        src: int,
        dst: int,
        max_length: int = 10000,
    ) -> Optional[np.ndarray]:
        """
        Backtrace path from destination to source using parent array.

        Args:
            parent: Parent array from Dijkstra
            src: Source node index
            dst: Destination node index
            max_length: Maximum path length to prevent infinite loops

        Returns:
            Path as array of node indices from src to dst, or None if no path
        """
        if parent[dst] == -1 and dst != src:
            return None

        path = [dst]
        current = dst

        while current != src and len(path) < max_length:
            p = parent[current]
            if p == -1 or p == current:
                return None  # No valid path
            path.append(p)
            current = p

        if current != src:
            return None

        return np.array(path[::-1], dtype=np.int32)


class NumpyViaKernels:
    """NumPy implementations of via-related CUDA kernels."""

    def apply_hard_blocks(
        self,
        edge_costs: np.ndarray,
        blocked_edges: np.ndarray,
        block_cost: float = 1e9,
    ) -> np.ndarray:
        """
        Apply hard blocks to edges by setting high costs.

        Args:
            edge_costs: Array of edge costs (modified in-place)
            blocked_edges: Boolean array or indices of blocked edges
            block_cost: Cost to assign to blocked edges

        Returns:
            Modified edge costs array
        """
        if blocked_edges.dtype == bool:
            edge_costs[blocked_edges] = block_cost
        else:
            edge_costs[blocked_edges] = block_cost
        return edge_costs

    def apply_via_penalties(
        self,
        edge_costs: np.ndarray,
        via_edges: np.ndarray,
        via_penalty: float,
    ) -> np.ndarray:
        """
        Apply via penalties to via edges.

        Args:
            edge_costs: Array of edge costs (modified in-place)
            via_edges: Boolean array or indices of via edges
            via_penalty: Penalty to add to via edges

        Returns:
            Modified edge costs array
        """
        if via_edges.dtype == bool:
            edge_costs[via_edges] += via_penalty
        else:
            edge_costs[via_edges] += via_penalty
        return edge_costs

    def detect_barrel_conflicts(
        self,
        via_positions: np.ndarray,
        clearance: float = 0.4,
    ) -> np.ndarray:
        """
        Detect barrel (via-to-via) conflicts.

        Args:
            via_positions: (N, 2) array of (x, y) via positions
            clearance: Minimum clearance between vias

        Returns:
            (M, 2) array of conflicting via pairs (indices)
        """
        N = len(via_positions)
        conflicts = []

        for i in range(N):
            for j in range(i + 1, N):
                dx = via_positions[i, 0] - via_positions[j, 0]
                dy = via_positions[i, 1] - via_positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < clearance:
                    conflicts.append([i, j])

        return np.array(conflicts, dtype=np.int32).reshape(-1, 2) if conflicts else np.zeros((0, 2), dtype=np.int32)


class NumpyROIKernels:
    """NumPy implementations of ROI extraction CUDA kernels."""

    def extract_roi_nodes(
        self,
        src: int,
        dst: int,
        Nx: int,
        Ny: int,
        Nz: int,
        margin: int = 10,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
        """
        Extract ROI (Region of Interest) nodes around source and destination.

        Args:
            src: Source node index
            dst: Destination node index
            Nx, Ny, Nz: Lattice dimensions
            margin: Margin around bounding box

        Returns:
            Tuple of (roi_node_indices, bbox)
            bbox is (minx, maxx, miny, maxy, minz, maxz)
        """
        plane_size = Nx * Ny

        # Convert to coordinates
        src_z = src // plane_size
        src_rem = src % plane_size
        src_y = src_rem // Nx
        src_x = src_rem % Nx

        dst_z = dst // plane_size
        dst_rem = dst % plane_size
        dst_y = dst_rem // Nx
        dst_x = dst_rem % Nx

        # Compute bounding box with margin
        minx = max(0, min(src_x, dst_x) - margin)
        maxx = min(Nx - 1, max(src_x, dst_x) + margin)
        miny = max(0, min(src_y, dst_y) - margin)
        maxy = min(Ny - 1, max(src_y, dst_y) + margin)
        minz = max(0, min(src_z, dst_z))
        maxz = min(Nz - 1, max(src_z, dst_z))

        # Generate all nodes in ROI
        roi_nodes = []
        for z in range(minz, maxz + 1):
            for y in range(miny, maxy + 1):
                for x in range(minx, maxx + 1):
                    node = z * plane_size + y * Nx + x
                    roi_nodes.append(node)

        bbox = (minx, maxx, miny, maxy, minz, maxz)
        return np.array(roi_nodes, dtype=np.int32), bbox

    def create_roi_bitmap(
        self,
        roi_nodes: np.ndarray,
        max_node: int,
    ) -> np.ndarray:
        """
        Create bit-packed ROI bitmap.

        Args:
            roi_nodes: Array of node indices in ROI
            max_node: Maximum node index

        Returns:
            Bit-packed uint32 array
        """
        num_words = (max_node + 31) // 32
        bitmap = np.zeros(num_words, dtype=np.uint32)

        for node in roi_nodes:
            if 0 <= node < max_node:
                word_idx = node >> 5
                bit_idx = node & 31
                bitmap[word_idx] |= (1 << bit_idx)

        return bitmap


# Singleton instances for easy access
_dijkstra_kernels = None
_via_kernels = None
_roi_kernels = None


def get_dijkstra_kernels(lattice_dims: Optional[Tuple[int, int, int]] = None) -> NumpyDijkstraKernels:
    """Get singleton Dijkstra kernels instance."""
    global _dijkstra_kernels
    if _dijkstra_kernels is None:
        _dijkstra_kernels = NumpyDijkstraKernels(lattice_dims)
    elif lattice_dims is not None:
        _dijkstra_kernels.set_lattice_dims(*lattice_dims)
    return _dijkstra_kernels


def get_via_kernels() -> NumpyViaKernels:
    """Get singleton via kernels instance."""
    global _via_kernels
    if _via_kernels is None:
        _via_kernels = NumpyViaKernels()
    return _via_kernels


def get_roi_kernels() -> NumpyROIKernels:
    """Get singleton ROI kernels instance."""
    global _roi_kernels
    if _roi_kernels is None:
        _roi_kernels = NumpyROIKernels()
    return _roi_kernels
