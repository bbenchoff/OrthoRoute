"""
GraphBuilder Mixin - Extracted from UnifiedPathFinder

This module contains graph builder mixin functionality.
Part of the PathFinder routing algorithm refactoring.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from types import SimpleNamespace

# Prefer local light interfaces; fall back to monorepo types if available
try:
    from ....domain.models.board import Board as BoardLike, Pad
except Exception:  # pragma: no cover - plugin environment
    from ..types import BoardLike, Pad

logger = logging.getLogger(__name__)


class GraphBuilderMixin:
    """
    GraphBuilder functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

    def _init_gpu_buffers_once(self, N):
        """One-time GPU buffer allocation - no per-net allocs"""
        if getattr(self, "_gpu_bufs_inited", False) and getattr(self, 'dist_gpu', None) is not None and self.dist_gpu.size >= N:
            return
        import cupy as cp
        self.dist_gpu = cp.empty((N,), dtype=cp.float32)
        self.parent_gpu = cp.empty((N,), dtype=cp.int32)
        self.in_bucket = cp.empty((N,), dtype=cp.uint8)
        self._gpu_bufs_inited = True
        logger.debug(f"[GPU-BUFFERS] Initialized once for N={N}")


    def _reset_gpu_buffers(self, n):
        """Reset GPU buffers for ROI of size n using .fill() - much faster than allocation"""
        import cupy as cp
        self.dist_gpu[:n].fill(cp.inf)
        self.parent_gpu[:n].fill(-1)
        self.in_bucket[:n].fill(0)


    def _ensure_gpu_edge_buffers(self, E: int):
        """Ensure GPU edge buffers are properly sized"""
        if not self.use_gpu:
            return

        import cupy as cp
        # This is called after CPU arrays are built, so just create GPU mirrors
        gpu_attrs = [
            'edge_total_penalty', 'edge_dir_mask', 'edge_bottleneck_penalty',
            'edge_present_usage', 'edge_history', 'edge_capacity', 'edge_total_cost'
        ]

        for attr in gpu_attrs:
            cpu_arr = getattr(self, attr, None)
            if cpu_arr is not None and not hasattr(cpu_arr, 'get'):  # Not already on GPU
                setattr(self, attr, cp.asarray(cpu_arr))

        logger.debug(f"[GPU-BUFFERS] Created GPU mirrors for {len(gpu_attrs)} edge arrays (E={E})")


    def _populate_cpu_csr(self):
        """Ensure CPU CSR arrays are populated from live adjacency matrix"""
        if hasattr(self, 'adjacency_matrix') and self.adjacency_matrix is not None:
            if self.use_gpu:
                # Copy from GPU to CPU
                self.indptr_cpu = self.adjacency_matrix.indptr.get()
                self.indices_cpu = self.adjacency_matrix.indices.get()
                self.weights_cpu = self.adjacency_matrix.data.get()
            else:
                # Already on CPU
                self.indptr_cpu = self.adjacency_matrix.indptr
                self.indices_cpu = self.adjacency_matrix.indices
                self.weights_cpu = self.adjacency_matrix.data

            logger.debug(f"[CSR] Populated CPU arrays: {len(self.indices_cpu)} edges")
        else:
            logger.warning("[CSR] No adjacency_matrix available for CPU CSR population")


    def _assert_live_sizes(self):
        """Defensive check: assert live sizes before operations to catch mismatches early"""
        gs = getattr(self, "graph_state", None)

        # Resolve N with backfill
        N = getattr(gs, "lattice_node_count", None)
        if N is None:
            N = getattr(self, "lattice_node_count", 0)
            if gs is not None:
                gs.lattice_node_count = N

        # Resolve indices with null-safety
        indices = None
        if gs is not None:
            indices = getattr(gs, "indices_cpu", None)
        if indices is None:
            indices = getattr(self, "indices_cpu", None)

        if indices is None:
            logger.warning("[LIVE-SIZE] indices_cpu not available yet; skipping size checks")
            return

        E = len(indices)
        # Backfill gs for downstream code paths
        if gs is not None and getattr(gs, "indices_cpu", None) is None:
            gs.indices_cpu = indices

        # Node coordinate check
        if hasattr(self, 'node_coordinates_lattice') and self.node_coordinates_lattice is not None:
            coord_count = self.node_coordinates_lattice.shape[0]
            assert coord_count == N, f"[LIVE-SIZE] node coord / N mismatch: {coord_count} != {N}"

        # STRICT edge array checks - fail hard on any mismatch
        edge_arrays = ['edge_total_penalty', 'edge_total_cost', 'edge_present_usage',
                      'edge_history', 'edge_capacity', 'edge_dir_mask', 'edge_bottleneck_penalty']

        for arr_name in edge_arrays:
            if hasattr(self, arr_name):
                arr = getattr(self, arr_name)
                if arr is not None:
                    arr_len = len(arr)
                    assert arr_len == E, f"[LIVE-SIZE] CRITICAL: {arr_name} size {arr_len} != E {E} - this causes truncation!"

        logger.debug(f"[LIVE-SIZE] All sizes verified: N={N}, E={E} - no truncation possible")


    def _build_gpu_matrices(self):
        """Build sparse adjacency matrices with optimal GPU/CPU backend selection.

        Constructs CSR (Compressed Sparse Row) format adjacency matrices optimized
        for the current computational backend. Automatically selects GPU acceleration
        when available and beneficial, falling back to CPU for compatibility.

        Note:
            - Creates CSR format sparse matrices for efficient graph operations
            - Handles both GPU (CuPy) and CPU (SciPy) sparse matrix backends
            - Includes comprehensive edge weight and capacity integration
            - Optimizes memory layout for subsequent pathfinding operations
            - Validates matrix dimensions against node/edge counts
            - Essential preparation step for all graph algorithms
        """
        if not self.edges:
            logger.error("No edges to build matrices from")
            return

        # Extract edges as 1-D arrays
        rows_np = np.asarray([e[0] for e in self.edges], dtype=np.int32)
        cols_np = np.asarray([e[1] for e in self.edges], dtype=np.int32)
        data_np = np.asarray([e[2] for e in self.edges], dtype=np.float32)

        use_gpu_csr = bool(getattr(self, "use_gpu", False) and CUPY_AVAILABLE)

        if use_gpu_csr:
            # GPU: use cupyx.sparse.csr_matrix explicitly
            from cupyx.scipy import sparse as csp  # type: ignore
            rows_cp = cp.asarray(rows_np, dtype=cp.int32)
            cols_cp = cp.asarray(cols_np, dtype=cp.int32)
            data_cp = cp.asarray(data_np, dtype=cp.float32)
            self.adjacency_matrix = sp.csr_matrix((data_cp, (rows_cp, cols_cp)),
                                                   shape=(self.node_count, self.node_count))
            # CPU mirrors for invariants/checks
            self.indptr_cpu = self.adjacency_matrix.indptr.get()
            self.indices_cpu = self.adjacency_matrix.indices.get()
            self.weights_cpu = self.adjacency_matrix.data.get()
        else:
            # CPU: use SciPy csr_matrix explicitly
            from scipy.sparse import csr_matrix as scipyc_csr  # type: ignore
            self.adjacency_matrix = scipyc_csr((data_np, (rows_np, cols_np)),
                                               shape=(self.node_count, self.node_count))
            self.indptr_cpu = self.adjacency_matrix.indptr
            self.indices_cpu = self.adjacency_matrix.indices
            self.weights_cpu = self.adjacency_matrix.data
        
        # Update coordinate array with any new escape nodes (coordinate array was pre-initialized) 
        if self.node_coordinates is None or self.node_coordinates.shape[0] != self.node_count:
            logger.info(f"Rebuilding coordinate array: current={0 if self.node_coordinates is None else self.node_coordinates.shape[0]} vs needed={self.node_count}")
            coords = np.zeros((self.node_count, 3))
            for node_id, (x, y, layer, idx) in self.nodes.items():
                coords[idx] = [x, y, layer]
            self.node_coordinates = cp.array(coords) if self.use_gpu else coords
        else:
            logger.info(f"Using pre-initialized coordinate array with {self.node_coordinates.shape[0]} entries")
        
        # Initialize PathFinder state - DEVICE ARRAYS (GPU/CPU mode-aware)
        num_edges = len(self.edges)
        if self.use_gpu:
            # Device arrays for GPU ∆-stepping
            self.edge_capacity = cp.ones(num_edges, dtype=cp.float32)  # Capacity = 1 per edge
            self.edge_present_usage = cp.zeros(num_edges, dtype=cp.float32)  # Current iteration usage
            self.edge_history = cp.zeros(num_edges, dtype=cp.float32)  # Historical congestion
            
            # DEVICE-ONLY ROI EXTRACTION: Persistent scratch arrays for global→local mapping
            # Pre-allocate maximum-size scratch arrays to avoid per-ROI allocations
            max_roi_nodes = min(10000, self.node_count)  # Conservative upper bound

            # SURGICAL ENHANCEMENT: Add ROI size caps for safety
            roi_safety_cap = ROI_SAFETY_CAP
            max_roi_nodes = min(max_roi_nodes, roi_safety_cap)

            self.g2l_scratch = cp.full(self.node_count, -1, dtype=cp.int32)  # Global→Local ID mapping
            self.roi_node_buffer = cp.empty(max_roi_nodes, dtype=cp.int32)  # ROI node IDs
            self.roi_edge_src_buffer = cp.empty(max_roi_nodes * 8, dtype=cp.int32)  # Edge sources (8 neighbors avg)
            self.roi_edge_dst_buffer = cp.empty(max_roi_nodes * 8, dtype=cp.int32)  # Edge destinations
            self.roi_edge_cost_buffer = cp.empty(max_roi_nodes * 8, dtype=cp.float32)  # Edge costs

            # Store for defensive bounds checking
            self.max_roi_nodes = max_roi_nodes
            
            # CuPy Events for precise GPU timing instrumentation
            self.roi_start_event = cp.cuda.Event()
            self.roi_extract_event = cp.cuda.Event()
            self.roi_edges_event = cp.cuda.Event()
            self.roi_end_event = cp.cuda.Event()
            
            logger.info(f"DEVICE-ONLY ROI: Allocated persistent scratch arrays for up to {max_roi_nodes} nodes per ROI")
            self.edge_bottleneck_penalty = cp.zeros(num_edges, dtype=cp.float32)  # Precomputed penalties
            self.edge_dir_mask = cp.ones(num_edges, dtype=cp.uint8)  # Direction enforcement - legal by default
            self.edge_total_cost = cp.zeros(num_edges, dtype=cp.float32)  # Combined cost per iteration
            
            # LEGACY arrays for compatibility - will be removed
            self.congestion = self.edge_present_usage
            self.history_cost = self.edge_history
        else:
            # CPU fallback
            self.edge_capacity = np.ones(num_edges, dtype=np.float32)
            self.edge_present_usage = np.zeros(num_edges, dtype=np.float32)
            self.edge_history = np.zeros(num_edges, dtype=np.float32)
            self.edge_bottleneck_penalty = np.zeros(num_edges, dtype=np.float32)
            self.edge_dir_mask = np.ones(num_edges, dtype=np.uint8)  # Legal by default
            self.edge_total_cost = np.zeros(num_edges, dtype=np.float32)
            
            # LEGACY
            self.congestion = self.edge_present_usage
            self.history_cost = self.edge_history
        
        # PRECOMPUTE edge penalties once on GPU
        self._precompute_edge_penalties()
        
        # PRECOMPUTE reverse edge index once during lattice building (major optimization)
        self._build_reverse_edge_index_gpu()
        
        logger.info(f"Built GPU matrices: {self.node_count:,} nodes, {num_edges:,} edges")

        # 5. BUILD GPU SPATIAL INDEX for ultra-fast ROI extraction
        disable_gpu_roi = DISABLE_GPU_ROI
        if getattr(self, "use_gpu", False) and not disable_gpu_roi:
            logger.info("Building GPU spatial index for constant-time ROI extraction...")
            self._build_gpu_spatial_index()
        else:
            logger.info("Skipping GPU spatial index (GPU ROI disabled); CPU ROI path will be used.")
            self._build_gpu_spatial_index()  # Still build it, but will use CPU path inside

        # 6. SYNC EDGE ARRAYS TO LIVE CSR after finalization
        self._sync_edge_arrays_to_live_csr()

        # 6.5. BUILD CSR EDGE LOOKUP for authoritative accounting
        self._build_edge_lookup_from_csr()

        # Rebuild fast edge index for rip-up / DRC lookups
        self._edge_index = {}
        src = getattr(self, "csr_src_cpu", None) or getattr(self, "src_cpu", None)
        dst = getattr(self, "csr_dst_cpu", None) or getattr(self, "dst_cpu", None)
        if src is not None and dst is not None:
            try:
                for i in range(len(src)):
                    self._edge_index[(int(src[i]), int(dst[i]))] = i
            except Exception:
                logger.debug("[EDGE-INDEX] build skipped (non-CPU arrays)")

        # 7. INITIALIZE ROI CACHE for stable regions
        self._roi_cache = {}  # net_id -> cached ROI data
        self._dirty_tiles = set()  # Track regions that need ROI rebuild
    

    def _precompute_edge_penalties(self):
        """Precompute bottleneck and direction penalties on GPU"""
        logger.info("Precomputing edge penalties on GPU...")
        
        if not self.use_gpu:
            return  # Skip for CPU
        
        # Get edge base costs as device array
        edge_base_costs = cp.array([edge[2] for edge in self.edges], dtype=cp.float32)
        
        # Precompute bottleneck penalty - vectorized on GPU
        edge_indices = cp.array([edge[1] for edge in self.edges], dtype=cp.int32)  # Target node indices
        edge_coords = self.node_coordinates[edge_indices]  # (N_edges, 3) - target coords
        
        # Board center and width for bottleneck detection
        board_center_x = (self.node_coordinates[:, 0].min() + self.node_coordinates[:, 0].max()) / 2
        board_width = self.node_coordinates[:, 0].max() - self.node_coordinates[:, 0].min()
        bottleneck_radius = board_width * 0.1  # 10% of board width
        
        # Vectorized bottleneck penalty: 2.0x cost for center channel
        center_distance = cp.abs(edge_coords[:, 0] - board_center_x)
        self.edge_bottleneck_penalty = cp.where(center_distance < bottleneck_radius, 2.0, 0.0)
        
        # NO direction mask needed - illegal edges were never created
        self.edge_dir_mask = cp.ones(len(self.edges), dtype=cp.uint8)  # All edges legal by default
        
        # Count bottleneck edges without host-device sync - use estimate
        logger.info(f"Precomputed penalties: edge penalties applied to center channel")
    

    def _build_edge_lookup_from_csr(self):
        """Build authoritative edge lookup table from CSR matrix representation.

        Creates a fast lookup table mapping (source, destination) node pairs to
        their corresponding edge indices in the CSR sparse matrix. This provides
        O(1) edge access and eliminates coordinate-based lookup drift issues.

        Note:
            - Replaces unreliable floating-point coordinate-based edge keys
            - Provides exact (u,v) -> edge_index mapping for graph traversal
            - Essential for PathFinder congestion tracking and cost updates
            - Uses CSR matrix as the single source of truth for edge existence
            - Eliminates numerical precision issues in coordinate-based lookups
            - Critical for maintaining graph consistency during routing
        """
        import numpy as np

        logger.info("[CSR-LOOKUP] Building edge lookup from CSR arrays...")

        # Get CSR structure (using correct attribute names from _build_gpu_matrices)
        if not hasattr(self, 'indices_cpu') or not hasattr(self, 'indptr_cpu'):
            logger.warning("[CSR-LOOKUP] CSR arrays not available, skipping edge lookup build")
            return

        # Initialize edge lookup and ownership tracking
        self.edge_lookup = {}  # (u,v) -> edge_index
        self.edge_owners = {}  # edge_index -> Set[str] (current owners)
        self.edge_usage_count = {}  # edge_index -> usage count

        # Build lookup from CSR structure (using correct attribute names)
        edge_count = 0
        for u in range(len(self.indptr_cpu) - 1):
            start_idx = self.indptr_cpu[u]
            end_idx = self.indptr_cpu[u + 1]

            for edge_idx in range(start_idx, end_idx):
                v = self.indices_cpu[edge_idx]

                # Store both directions for undirected graph
                self.edge_lookup[(u, v)] = edge_idx
                self.edge_lookup[(v, u)] = edge_idx  # Symmetric access

                # Initialize edge accounting
                self.edge_owners[edge_idx] = set()  # No current owner(s) yet
                self.edge_usage_count[edge_idx] = 0  # No current usage

                edge_count += 1

        # Track size for consistency checks
        self._edge_lookup_size = self.edge_present_usage.shape[0]
        logger.info(f"[CSR-LOOKUP] Built edge lookup: {len(self.edge_lookup)} (E_live={self._edge_lookup_size})")
        logger.info(f"[CSR-LOOKUP] Initialized {len(self.edge_owners)} edge ownership records")


    def _sync_edge_arrays_to_live_csr(self):
        """
        Ensure ALL edge-dependent arrays match the live CSR edge count (E_live).
        Safe to call repeatedly; idempotent. Creates missing arrays with sane defaults.
        """
        import numpy as np
        E_live = self._live_edge_count()

        # Pull CPU CSR weights if available (best base for costs)
        weights = getattr(self, "weights_cpu", None)
        if weights is not None and len(weights) != E_live:
            # If weights are stale, rebuild from adjacency if you have a builder;
            # otherwise pad with last value (or 1.0) to avoid crashes.
            if len(weights) < E_live:
                pad = np.full(E_live - len(weights), float(weights[-1]) if len(weights) else 1.0, dtype=np.float32)
                self.weights_cpu = np.concatenate([weights.astype(np.float32, copy=False), pad])
            else:
                self.weights_cpu = weights[:E_live].astype(np.float32, copy=False)

        def _ensure_len(name, default_value, dtype):
            arr = getattr(self, name, None)
            if arr is None:
                setattr(self, name, np.full(E_live, default_value, dtype=dtype))
                return
            if len(arr) == E_live:
                # normalize dtype
                if arr.dtype != np.dtype(dtype):
                    setattr(self, name, arr.astype(dtype, copy=False))
                return
            if len(arr) < E_live:
                pad = np.full(E_live - len(arr), default_value, dtype=dtype)
                new_arr = np.concatenate([arr.astype(dtype, copy=False), pad])
            else:
                new_arr = arr[:E_live].astype(dtype, copy=False)
            setattr(self, name, new_arr)

        # Edge arrays you use in cost math / masks / accounting
        _ensure_len("edge_total_penalty", 0.0, np.float32)   # penalties added on top
        _ensure_len("edge_bottleneck_penalty", 0.0, np.float32)
        _ensure_len("edge_present_usage", 0.0, np.float32)   # negotiated congestion
        _ensure_len("edge_history", 0.0, np.float32)         # historical congestion
        _ensure_len("edge_capacity", 1.0, np.float32)        # capacity (if used)
        _ensure_len("edge_dir_mask", 1, np.uint8)            # 0/1 legal mask
        _ensure_len("edge_total_cost", 0.0, np.float32)      # output buffer
        _ensure_len("edge_base_cost", 0.0, np.float32)       # if you keep a cached base

        # If you don't maintain edge_base_cost separately, synthesize from weights + static penalties:
        if getattr(self, "edge_base_cost", None) is None and getattr(self, "weights_cpu", None) is not None:
            w = self.weights_cpu.astype(np.float32, copy=False)
            pen = self.edge_total_penalty.astype(np.float32, copy=False)
            self.edge_base_cost = (w[:E_live] + pen[:E_live]).astype(np.float32, copy=False)

        # Mirror to GPU if needed
        if getattr(self, "use_gpu", False):
            try:
                import cupy as cp
                for name in (
                    "edge_total_penalty", "edge_bottleneck_penalty", "edge_present_usage",
                    "edge_history", "edge_capacity", "edge_dir_mask", "edge_total_cost", "edge_base_cost"
                ):
                    cpu_arr = getattr(self, name)
                    if not hasattr(cpu_arr, "get"):  # not a CuPy array
                        setattr(self, name, cp.asarray(cpu_arr))
            except Exception:
                # If CuPy unavailable, stay CPU
                pass

        logger.info(f"[LIVE-SIZE] Edge arrays synced to E_live={E_live} (no truncation/mismatch)")

        # Keep CSR lookup in sync to prevent index space mismatches
        self._build_edge_lookup_from_csr()


    def _disable_incident_track_edges(self, z: int, x: int, y: int, keepout_weight: float) -> int:
        """
        For the grid node (x,y,z), set an effectively infinite weight on any
        incident H/V track edges so no trace may pass that point.
        """
        disabled = 0
        u = self.lattice.node_idx(x, y, z)
        # gather neighbor nodes in H/V directions (respect lattice bounds)
        nbrs = []
        for dx, dy in ((+1, 0), (-1, 0), (0, +1), (0, -1)):
            nx, ny = x + dx, y + dy
            if self.lattice.in_bounds(nx, ny):
                nbrs.append(self.lattice.node_idx(nx, ny, z))
        for v in nbrs:
            # edge_lookup stores both (u,v) and (v,u)
            idx = self.edge_lookup.get((u, v))
            if idx is None:
                continue
            # Identify track edges by layer discipline (same z) — only block tracks
            # (via edges are z1!=z2 and not touched here)
            self.graph_state["weights_cpu"][idx] = keepout_weight
            # also block reverse edge to maintain symmetry
            r_idx = self.edge_lookup.get((v, u))
            if r_idx is not None:
                self.graph_state["weights_cpu"][r_idx] = keepout_weight
            disabled += 1 + (1 if r_idx is not None else 0)
        return disabled


