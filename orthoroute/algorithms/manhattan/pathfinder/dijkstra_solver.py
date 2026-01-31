"""
Unified Dijkstra Solver

Provides a cross-platform Dijkstra shortest path solver that automatically
selects the best available backend:
1. CUDA (via CuPy) - for NVIDIA GPUs
2. NumPy - for CPU fallback (works on all platforms including Apple Silicon)

This module wraps the CUDA-specific implementation and provides a NumPy
fallback for systems without NVIDIA GPUs.
"""
import logging
from typing import List, Optional, Tuple, Any
import numpy as np

from ....infrastructure.gpu.compute_backend import get_backend, BackendType
from ....infrastructure.gpu.numpy_kernels import get_dijkstra_kernels

logger = logging.getLogger(__name__)


# Check CUDA availability
try:
    from .cuda_dijkstra import CUDADijkstra, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    CUDADijkstra = None


class NumpyDijkstraSolver:
    """
    NumPy-based Dijkstra solver for CPU execution.

    This provides the same interface as CUDADijkstra but runs on CPU
    using NumPy for cross-platform compatibility.
    """

    def __init__(self, graph=None, lattice=None):
        """Initialize NumPy Dijkstra solver."""
        self.indptr = None
        self.indices = None

        if graph:
            self.indptr = np.asarray(graph.indptr) if hasattr(graph.indptr, 'get') else graph.indptr
            self.indices = np.asarray(graph.indices) if hasattr(graph.indices, 'get') else graph.indices

        self.lattice = lattice
        self._kernels = get_dijkstra_kernels()

        if lattice:
            self._kernels.set_lattice_dims(
                lattice.x_steps if hasattr(lattice, 'x_steps') else lattice['x_steps'],
                lattice.y_steps if hasattr(lattice, 'y_steps') else lattice['y_steps'],
                lattice.num_layers if hasattr(lattice, 'num_layers') else lattice['num_layers'],
            )

    def find_paths_on_rois(
        self,
        roi_batch: List[Tuple],
        use_bitmap: bool = True
    ) -> List[Optional[List[int]]]:
        """
        Find paths on ROI subgraphs using CPU-based Dijkstra.

        Args:
            roi_batch: List of ROI tuples with graph data
            use_bitmap: Whether to enforce ROI bitmap boundaries

        Returns:
            List of paths (local ROI indices), one per ROI
        """
        if not roi_batch:
            return []

        K = len(roi_batch)
        logger.info(f"[NumPy-ROI] Processing {K} ROI subgraphs using CPU Dijkstra")

        paths = []
        for roi_idx, roi in enumerate(roi_batch):
            try:
                path = self._solve_single_roi(roi, use_bitmap)
                paths.append(path)
            except Exception as e:
                logger.warning(f"[NumPy-ROI] ROI {roi_idx} failed: {e}")
                paths.append(None)

        found = sum(1 for p in paths if p)
        logger.info(f"[NumPy-ROI] Complete: {found}/{K} paths found using CPU")
        return paths

    def _solve_single_roi(
        self,
        roi: Tuple,
        use_bitmap: bool = True
    ) -> Optional[List[int]]:
        """
        Solve Dijkstra on a single ROI using NumPy.

        Args:
            roi: ROI tuple with graph data
            use_bitmap: Whether to enforce bitmap

        Returns:
            Path as list of node indices, or None
        """
        # Normalize tuple length
        roi = self._normalize_roi_tuple(roi)

        # Extract ROI data (13-element tuple format)
        # Format: (roi_nodes, g2r, bbox, entry_layer, exit_layer, src, dst,
        #          via_mask, plane_size, xs, ys, zmin, zmax)
        # For simpler 6-element format: (indptr, indices, weights, size, src, dst)

        if len(roi) == 13:
            src = roi[5]
            dst = roi[6]
            # For 13-element, graph data is often in indices 2-4
            indptr = np.asarray(roi[2]) if not isinstance(roi[2], np.ndarray) else roi[2]
            indices = np.asarray(roi[3]) if not isinstance(roi[3], np.ndarray) else roi[3]
            weights = np.asarray(roi[4]) if not isinstance(roi[4], np.ndarray) else roi[4]
            roi_size = roi[8] if len(roi) > 8 and roi[8] is not None else len(indptr) - 1
        elif len(roi) >= 6:
            # Older format: (src, dst, indptr, indices, weights, roi_size, ...)
            src = roi[0]
            dst = roi[1]
            indptr = np.asarray(roi[2]) if hasattr(roi[2], 'get') else np.asarray(roi[2])
            indices = np.asarray(roi[3]) if hasattr(roi[3], 'get') else np.asarray(roi[3])
            weights = np.asarray(roi[4]) if hasattr(roi[4], 'get') else np.asarray(roi[4])
            roi_size = roi[5]
        else:
            logger.warning(f"Unsupported ROI tuple format with {len(roi)} elements")
            return None

        # Initialize distance and parent arrays
        dist = np.full(roi_size, np.inf, dtype=np.float32)
        parent = np.full(roi_size, -1, dtype=np.int32)
        dist[src] = 0.0

        # Create initial frontier
        frontier = np.zeros(roi_size, dtype=bool)
        frontier[src] = True

        # Extract bitmap if available
        roi_bitmap = None
        if use_bitmap and len(roi) >= 7 and roi[6] is not None:
            bitmap_data = roi[6]
            if hasattr(bitmap_data, 'get'):
                roi_bitmap = bitmap_data.get()
            elif isinstance(bitmap_data, np.ndarray):
                roi_bitmap = bitmap_data

        # Run wavefront expansion until convergence
        max_iterations = roi_size + 100  # Safety limit
        for iteration in range(max_iterations):
            new_frontier = self._kernels.wavefront_expand(
                frontier=frontier,
                indptr=indptr,
                indices=indices,
                total_cost=weights,
                dist=dist,
                parent=parent,
                roi_bitmap=roi_bitmap,
                use_bitmap=use_bitmap and roi_bitmap is not None,
            )

            # Check for convergence
            if not new_frontier.any():
                break

            frontier = new_frontier

            # Early exit if destination reached
            if dist[dst] < np.inf:
                # Continue a few more iterations for potential better paths
                pass

        # Backtrace path
        if dist[dst] == np.inf:
            return None

        return self._kernels.backtrace_path(parent, src, dst)

    def _normalize_roi_tuple(self, t: Tuple) -> Tuple:
        """Normalize ROI tuple to expected format."""
        if len(t) == 13:
            return t
        if len(t) == 11:
            roi_nodes, g2r, bbox, src, dst, via_mask, plane, xs, ys, zmin, zmax = t
            entry_layer = zmin
            exit_layer = zmax
            return (roi_nodes, g2r, bbox, entry_layer, exit_layer, src, dst,
                    via_mask, plane, xs, ys, zmin, zmax)
        if len(t) == 6:
            # Minimal format: (src, dst, indptr, indices, weights, roi_size)
            return t  # Keep as-is for simple format
        return t

    def find_path_batch(
        self,
        adjacency_csr,
        edge_costs: np.ndarray,
        sources: List[int],
        sinks: List[int],
        max_iterations: int = 1_000_000
    ) -> List[Optional[List[int]]]:
        """
        Find shortest paths for multiple source/sink pairs.

        Args:
            adjacency_csr: CSR adjacency matrix
            edge_costs: Edge costs array
            sources: List of source node indices
            sinks: List of sink node indices
            max_iterations: Maximum iterations per search

        Returns:
            List of paths (each path is list of node indices, or None)
        """
        # Convert CSR to numpy if needed
        if hasattr(adjacency_csr, 'get'):
            indptr = adjacency_csr.indptr.get()
            indices = adjacency_csr.indices.get()
        else:
            indptr = np.asarray(adjacency_csr.indptr)
            indices = np.asarray(adjacency_csr.indices)

        if hasattr(edge_costs, 'get'):
            edge_costs = edge_costs.get()
        else:
            edge_costs = np.asarray(edge_costs)

        num_pairs = len(sources)
        num_nodes = len(indptr) - 1

        logger.info(f"[NumPy] Batch Dijkstra: {num_pairs} paths on {num_nodes} nodes")

        paths = []
        for i, (src, dst) in enumerate(zip(sources, sinks)):
            path = self._dijkstra_single(indptr, indices, edge_costs, src, dst, num_nodes, max_iterations)
            paths.append(path)

        found = sum(1 for p in paths if p)
        logger.info(f"[NumPy] Batch complete: {found}/{num_pairs} paths found")
        return paths

    def _dijkstra_single(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        source: int,
        sink: int,
        num_nodes: int,
        max_iterations: int
    ) -> Optional[List[int]]:
        """Run standard Dijkstra for a single source/sink pair."""
        import heapq

        dist = np.full(num_nodes, np.inf, dtype=np.float32)
        parent = np.full(num_nodes, -1, dtype=np.int32)
        dist[source] = 0.0

        # Priority queue: (distance, node)
        pq = [(0.0, source)]
        visited = np.zeros(num_nodes, dtype=bool)

        iterations = 0
        while pq and iterations < max_iterations:
            iterations += 1
            d, u = heapq.heappop(pq)

            if visited[u]:
                continue
            visited[u] = True

            if u == sink:
                break

            # Relax edges
            start = indptr[u]
            end = indptr[u + 1]

            for edge_idx in range(start, end):
                v = indices[edge_idx]
                w = weights[edge_idx]
                new_dist = d + w

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Backtrace
        if dist[sink] == np.inf:
            return None

        path = []
        curr = sink
        while curr != -1:
            path.append(curr)
            prev = parent[curr]
            if prev == curr:
                break
            curr = prev

        path.reverse()
        return path if path and path[0] == source else None

    def find_path_single(
        self,
        adjacency_csr,
        edge_costs,
        source: int,
        sink: int,
        max_iterations: int = 1_000_000
    ) -> Optional[List[int]]:
        """Find single shortest path."""
        paths = self.find_path_batch(adjacency_csr, edge_costs, [source], [sink], max_iterations)
        return paths[0] if paths else None


class DijkstraSolver:
    """
    Unified Dijkstra solver that automatically selects the best backend.

    This class provides a consistent interface regardless of whether
    CUDA is available. It will use:
    - CUDADijkstra when NVIDIA GPU is available
    - NumpyDijkstraSolver otherwise (works on Apple Silicon, etc.)
    """

    def __init__(self, graph=None, lattice=None, force_cpu: bool = False):
        """
        Initialize Dijkstra solver.

        Args:
            graph: Optional graph object with indptr/indices
            lattice: Optional lattice configuration
            force_cpu: If True, always use CPU even if GPU is available
        """
        self._use_cuda = False
        self._solver = None

        backend = get_backend()

        if not force_cpu and CUDA_AVAILABLE and backend.backend_type == BackendType.CUPY:
            try:
                self._solver = CUDADijkstra(graph, lattice)
                self._use_cuda = True
                logger.info("[DijkstraSolver] Using CUDA backend")
            except Exception as e:
                logger.warning(f"[DijkstraSolver] CUDA init failed: {e}, falling back to NumPy")
                self._solver = NumpyDijkstraSolver(graph, lattice)
        else:
            self._solver = NumpyDijkstraSolver(graph, lattice)
            logger.info("[DijkstraSolver] Using NumPy backend")

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA backend."""
        return self._use_cuda

    def find_paths_on_rois(
        self,
        roi_batch: List[Tuple],
        use_bitmap: bool = True
    ) -> List[Optional[List[int]]]:
        """Find paths on ROI subgraphs."""
        return self._solver.find_paths_on_rois(roi_batch, use_bitmap)

    def find_path_batch(
        self,
        adjacency_csr,
        edge_costs,
        sources: List[int],
        sinks: List[int],
        max_iterations: int = 1_000_000
    ) -> List[Optional[List[int]]]:
        """Find shortest paths for multiple source/sink pairs."""
        return self._solver.find_path_batch(adjacency_csr, edge_costs, sources, sinks, max_iterations)

    def find_path_single(
        self,
        adjacency_csr,
        edge_costs,
        source: int,
        sink: int,
        max_iterations: int = 1_000_000
    ) -> Optional[List[int]]:
        """Find single shortest path."""
        return self._solver.find_path_single(adjacency_csr, edge_costs, source, sink, max_iterations)


def get_solver(graph=None, lattice=None, force_cpu: bool = False) -> DijkstraSolver:
    """
    Get a Dijkstra solver instance.

    This is the recommended way to get a solver, as it handles
    backend selection automatically.

    Args:
        graph: Optional graph object
        lattice: Optional lattice configuration
        force_cpu: Force CPU backend even if GPU available

    Returns:
        DijkstraSolver instance
    """
    return DijkstraSolver(graph, lattice, force_cpu)
