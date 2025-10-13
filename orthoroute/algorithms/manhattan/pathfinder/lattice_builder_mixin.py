"""
LatticeBuilder Mixin - Extracted from UnifiedPathFinder

This module contains lattice builder mixin functionality.
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
from ....domain.models.board import Board, Pad

logger = logging.getLogger(__name__)


class LatticeBuilderMixin:
    """
    LatticeBuilder functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

    def build_routing_lattice(self, board: Board) -> bool:
        """
        OPTIMIZED lattice building with spatial indexing
        Replaces both FastLatticeBuilder and LatticeBuilder
        """
        logger.info("Building optimized routing lattice...")
        start_time = time.time()
        
        # 1. Fast bounds calculation
        bounds_tuple = self._calculate_bounds_fast(board)
        min_x, min_y, max_x, max_y = bounds_tuple
        
        # Create proper Bounds object for spatial indexing
        from ...domain.models.board import Bounds
        self._board_bounds = Bounds(min_x, min_y, max_x, max_y)
        
        logger.info(f"Board bounds: ({min_x}, {min_y}, {max_x}, {max_y})")
        
        # 2. Build 3D lattice with dynamic layer count from KiCad
        layers = int(board.layer_count)  # Use actual KiCad layer count (user set to 10)
        assert layers >= 2, f"Need at least 2 copper layers, got {layers}"
        self.layer_count = layers  # Store for ROI extraction
        logger.info(f"Using {layers} copper layers from KiCad stackup")

        # Extract layer names from KiCad for proper H/V polarity assignment
        self.config.layer_count = layers
        self.config.layer_names = self._get_standard_layer_names(layers)
        logger.info(f"Layer names: {self.config.layer_names}")

        self._build_3d_lattice(bounds_tuple, layers)
        
        # 3. CRITICAL FIX: Initialize coordinate array BEFORE escape routing  
        self._initialize_coordinate_array()
        
        # 4. OPTIMIZED pad connections with spatial indexing
        self._connect_pads_optimized(self._get_all_pads(board))
        
        # 4.1. ASSERT coordinate consistency after escape routing
        self._assert_coordinate_consistency()
        
        # 4. Convert to GPU matrices
        self._build_gpu_matrices()
        
        # 5. BUILD GPU SPATIAL INDEX for ultra-fast ROI extraction (AFTER matrices)
        self._build_gpu_spatial_index()
        
        # 6. INITIALIZE ROI CACHE for stable regions
        self._roi_cache = {}  # net_id -> cached ROI data
        self._dirty_tiles = set()  # Track regions that need ROI rebuild
        
        # 7. SETUP GPU STREAMS for ROI preparation overlap
        if self.use_gpu:
            try:

                self._roi_stream = cp.cuda.Stream()  # Dedicated stream for ROI extraction
                self._compute_stream = cp.cuda.Stream()  # Main compute stream
                logger.info("GPU streams initialized for ROI overlap processing")
            except Exception as e:
                logger.warning(f"GPU streams setup failed: {e}")
                self._roi_stream = None
                self._compute_stream = None
        
        build_time = time.time() - start_time
        logger.info(f"Optimized lattice built: {self.node_count:,} nodes, {len(self.edges):,} edges in {build_time:.2f}s")
        
        # CRITICAL: Validate spatial integrity after escape routing
        if not self._validate_spatial_integrity():
            logger.error("Spatial integrity check failed - rebuilding spatial index")
            self._build_gpu_spatial_index()

        # Apply CSR masks after CSR matrix is fully built
        self._apply_csr_masks(board)

        # SURGICAL: Create canonical graph_state at the end of initialize_graph(board)
        # First assign node_coordinates_lattice for compatibility
        self.node_coordinates_lattice = getattr(self, 'node_coordinates', None)
        if self.node_coordinates_lattice is not None:
            self.lattice_node_count = int(self.node_coordinates_lattice.shape[0])
        else:
            self.lattice_node_count = getattr(self, 'node_count', 0)

        self.graph_state = SimpleNamespace(
            lattice_node_count = self.lattice_node_count,
            node_coordinates_lattice = self.node_coordinates_lattice,  # N×3 (x,y,layer)
            indptr_cpu   = getattr(self, 'indptr_cpu', None),
            indices_cpu  = getattr(self, 'indices_cpu', None),
            weights_cpu  = getattr(self, 'weights_cpu', None),
            rev_index    = getattr(self, '_reverse_edge_index', None),
        )

        # Fingerprint so we never regress silently:
        edge_count = len(getattr(self, 'edges', [])) if hasattr(self, 'edges') else 0
        logger.info("[GS] ready: nodes=%d edges=%d",
                    self.graph_state.lattice_node_count,
                    edge_count)

        return True
    

    def _validate_spatial_integrity(self):
        """Validate spatial index integrity after escape routing"""
        ok = True
        if self.node_coordinates is None or self.node_coordinates.shape[0] != self.node_count:
            logger.error(f"coords rows {0 if self.node_coordinates is None else self.node_coordinates.shape[0]} "
                        f"!= node_count {self.node_count}")
            ok = False
        if self._spatial_indptr is None or self._spatial_node_ids is None:
            logger.error("spatial index missing")
            ok = False
        else:
            if self._spatial_indptr.ndim != 1 or self._spatial_indptr.size < 2:
                logger.error("indptr malformed")
                ok = False
            # lightweight CSR sanity
            if (self._spatial_indptr.dtype != cp.int32 or
                self._spatial_node_ids.dtype != cp.int32):
                logger.warning("casting spatial arrays to int32")
                self._spatial_indptr = self._spatial_indptr.astype(cp.int32, copy=False)
                self._spatial_node_ids = self._spatial_node_ids.astype(cp.int32, copy=False)
        return ok
    

    def _calculate_bounds_fast(self, board: Board) -> Tuple[float, float, float, float]:
        """Fast bounds calculation with airwire-constrained routing area"""

        # ENHANCEMENT: Calculate airwire bounding box + margin for efficient routing
        ROUTING_MARGIN = 3.0  # mm - margin around airwires for routing area

        # First, try to get airwire bounds for constrained routing
        airwire_bounds = self._calculate_airwire_bounds(board)
        if airwire_bounds:
            min_x, min_y, max_x, max_y = airwire_bounds
            # Add margin around airwires
            min_x -= ROUTING_MARGIN
            min_y -= ROUTING_MARGIN
            max_x += ROUTING_MARGIN
            max_y += ROUTING_MARGIN
            logger.info(f"[BOUNDS] Using airwire bounds + {ROUTING_MARGIN}mm margin: ({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")
            return (min_x, min_y, max_x, max_y)

        # PRIORITY: Use pad-based bounds (tighter) over full board bounds
        # Try to calculate from actual pad positions first
        try:
            all_pads = self._get_all_pads(board)
            if all_pads and len(all_pads) > 0:
                all_x = [self._get_pad_coordinates(pad)[0] for pad in all_pads]
                all_y = [self._get_pad_coordinates(pad)[1] for pad in all_pads]
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                # Add routing margin
                min_x -= ROUTING_MARGIN
                min_y -= ROUTING_MARGIN
                max_x += ROUTING_MARGIN
                max_y += ROUTING_MARGIN
                logger.info(f"[BOUNDS] Using pad-based bounds + {ROUTING_MARGIN}mm margin: ({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")
                logger.info(f"[BOUNDS] Covers {len(all_pads)} pads (tighter than full board bounds)")
                return (min_x, min_y, max_x, max_y)
        except Exception as e:
            logger.warning(f"[BOUNDS] Pad-based calculation failed: {e}")
        
        # Fallback: Use KiCad bounds only if pads unavailable
        if hasattr(board, '_kicad_bounds'):
            kicad_bounds = board._kicad_bounds
            min_x, min_y, max_x, max_y = kicad_bounds
            # Add routing margin around KiCad bounds
            min_x -= ROUTING_MARGIN  
            min_y -= ROUTING_MARGIN
            max_x += ROUTING_MARGIN
            max_y += ROUTING_MARGIN
            logger.warning(f"[BOUNDS] Using FULL BOARD KiCad bounds + {ROUTING_MARGIN}mm margin (fallback): ({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})")
            logger.warning(f"[BOUNDS] WARNING: Grid will cover entire board, not just routing area!")
            return (min_x, min_y, max_x, max_y)

        # Final fallback
        logger.error(f"[BOUNDS] All bounds calculation methods failed, using default")
        return (0, 0, 100, 100)
    

    def _build_3d_lattice(self, bounds: Tuple[float, float, float, float], layers: int):
        """Build optimized 3D routing lattice with layer-aware connectivity.

        Constructs the complete 3D routing graph using KiCadGeometry as the
        authoritative source for layer definitions, constraints, and routing rules.

        Args:
            bounds (Tuple[float, float, float, float]): Board bounds as (min_x, min_y, max_x, max_y) in mm
            layers (int): Number of routing layers to create

        Note:
            - Initializes KiCadGeometry system with dynamic layer count
            - Sets up H/V polarity with F.Cu vertical routing by requirement
            - Initializes occupancy grids for DRC (Design Rule Check)
            - Creates R-tree spatial indices for efficient clearance checking
            - Configures PathFinder edge tracking with congestion accounting
            - Performs comprehensive sanity checks and parameter validation
            - Foundation for all subsequent pathfinding operations
        """
        # Initialize KiCad-based geometry system with dynamic layer count
        self.geometry = KiCadGeometry(bounds, self.config.grid_pitch)
        self.geometry.layer_count = layers

        # Set up HV polarity with F.Cu vertical by requirement
        self.geometry.layer_directions = self._make_hv_polarity(self.config.layer_names)
        logger.info(f"Using {layers} layers with HV polarity: {self.geometry.layer_directions}")

        # Log detailed polarity mapping
        for i, (name, direction) in enumerate(zip(self.config.layer_names, self.geometry.layer_directions)):
            logger.info(f"  Layer {i}: {name} = {direction.upper()}")

        # Store layer configuration for router
        self.hv_polarity = self.geometry.layer_directions

        # Initialize occupancy grids for DRC (one per layer)
        self._init_occupancy_grids(layers)

        # Initialize R-tree spatial indices for clearance checking
        self._initialize_layer_rtrees(layers)

        # Initialize PathFinder edge tracking with proper congestion accounting
        self._init_pathfinder_edge_tracking()

        # SANITY CHECKS/LOGS: Log critical build parameters
        self._log_build_sanity_checks(layers)

        logger.info(f"KiCad bounds: {bounds}")
        logger.info(f"Grid aligned: ({self.geometry.grid_min_x}, {self.geometry.grid_min_y}) to ({self.geometry.grid_max_x}, {self.geometry.grid_max_y})")
        logger.info(f"3D lattice: {self.geometry.x_steps} x {self.geometry.y_steps} x {layers} = {self.geometry.x_steps * self.geometry.y_steps * layers:,} nodes")

        # Create nodes using KiCadGeometry
        edges = []

        for layer in range(layers):
            direction = self.geometry.layer_directions[layer]
            layer_nodes = []

            # Create nodes for this layer using geometry system
            for x_idx in range(self.geometry.x_steps):
                for y_idx in range(self.geometry.y_steps):
                    world_x, world_y = self.geometry.lattice_to_world(x_idx, y_idx)
                    node_idx = self.geometry.node_index(x_idx, y_idx, layer)

                    node_id = f"rail_{direction}_{x_idx}_{y_idx}_{layer}"
                    self.nodes[node_id] = (world_x, world_y, layer, node_idx)
                    self._node_lookup[node_id] = node_idx
                    layer_nodes.append((world_x, world_y, node_id, node_idx))

            # Store spatial index for this layer
            self._spatial_index[layer] = layer_nodes

            # Create edges only for legal directions using geometry validation
            if layer == 0:
                max_trace_steps = 2  # F.Cu: only short escapes
                escape_cost = 1.0
            else:
                max_trace_steps = max(self.geometry.x_steps, self.geometry.y_steps)
                escape_cost = 1.0

            if direction == 'h':
                # H-layer: only horizontal edges
                for y_idx in range(self.geometry.y_steps):
                    for x_idx in range(min(self.geometry.x_steps - 1, max_trace_steps)):
                        from_idx = self.geometry.node_index(x_idx, y_idx, layer)
                        to_idx = self.geometry.node_index(x_idx + 1, y_idx, layer)

                        # Validate edge using geometry system
                        if self.geometry.is_valid_edge(x_idx, y_idx, layer, x_idx + 1, y_idx, layer):
                            edges.extend([(from_idx, to_idx, escape_cost * self.geometry.pitch),
                                        (to_idx, from_idx, escape_cost * self.geometry.pitch)])
            else:
                # V-layer: only vertical edges
                for x_idx in range(self.geometry.x_steps):
                    for y_idx in range(min(self.geometry.y_steps - 1, max_trace_steps)):
                        from_idx = self.geometry.node_index(x_idx, y_idx, layer)
                        to_idx = self.geometry.node_index(x_idx, y_idx + 1, layer)

                        # Validate edge using geometry system
                        if self.geometry.is_valid_edge(x_idx, y_idx, layer, x_idx, y_idx + 1, layer):
                            edges.extend([(from_idx, to_idx, escape_cost * self.geometry.pitch),
                                        (to_idx, from_idx, escape_cost * self.geometry.pitch)])

        # Create inter-layer via connections with configurable cost and legal transitions
        VIA_COST_LOCAL = VIA_COST  # Via cost penalty
        VIA_CAP_PER_NET = VIA_CAPACITY_PER_NET

        logger.info(f"Via configuration: cost={VIA_COST}, cap_per_net={VIA_CAP_PER_NET}")

        # Build legal layer transitions from KiCad stackup rules
        self.allowed_layer_pairs = self._derive_allowed_layer_pairs(layers)
        logger.info(f"Legal via transitions: {len(self.allowed_layer_pairs)} pairs")

        # Log first few transitions for debugging
        for i, (from_l, to_l) in enumerate(sorted(self.allowed_layer_pairs)):
            if i < 10:  # First 10
                from_name = self.config.layer_names[from_l] if from_l < len(self.config.layer_names) else f"L{from_l}"
                to_name = self.config.layer_names[to_l] if to_l < len(self.config.layer_names) else f"L{to_l}"
                logger.info(f"  {from_name} <-> {to_name}")
        if len(self.allowed_layer_pairs) > 10:
            logger.info(f"  ... and {len(self.allowed_layer_pairs)-10} more")

        # Create via edges for legal layer transitions only
        via_edges_created = 0
        for x_idx in range(self.geometry.x_steps):
            for y_idx in range(self.geometry.y_steps):
                for from_layer, to_layer in self.allowed_layer_pairs:
                    from_idx = self.geometry.node_index(x_idx, y_idx, from_layer)
                    to_idx = self.geometry.node_index(x_idx, y_idx, to_layer)
                    edges.extend([(from_idx, to_idx, VIA_COST), (to_idx, from_idx, VIA_COST)])
                    via_edges_created += 2

        logger.info(f"Created {via_edges_created:,} via edges (bidirectional) for legal transitions")

        self.edges = edges
        self.node_count = self.geometry.x_steps * self.geometry.y_steps * layers
        logger.info(f"Created {len(edges):,} edges")

        # Verify lattice correctness using geometry system
        self._verify_lattice_correctness_geometry()


    def _verify_lattice_correctness_geometry(self):
        """Verify lattice correctness using KiCadGeometry system"""
        logger.info("VERIFYING LATTICE CORRECTNESS (KiCad-based)...")

        illegal_edges = 0
        total_edges_checked = 0

        # H/V discipline counters per layer
        layer_h_edges = {}  # layer_id -> count of horizontal edges
        layer_v_edges = {}  # layer_id -> count of vertical edges

        for layer_id in range(self.geometry.layer_count):
            layer_h_edges[layer_id] = 0
            layer_v_edges[layer_id] = 0

        # Check all edges using geometry system
        for from_idx, to_idx, cost in self.edges:
            # Convert node indices back to coordinates
            from_x, from_y, from_layer = self.geometry.index_to_coords(from_idx)
            to_x, to_y, to_layer = self.geometry.index_to_coords(to_idx)

            # Skip via connections
            if from_layer != to_layer:
                continue

            total_edges_checked += 1

            # Count H/V edges per layer
            is_horizontal = (from_y == to_y and from_x != to_x)
            is_vertical = (from_x == to_x and from_y != to_y)

            if is_horizontal:
                layer_h_edges[from_layer] += 1
            elif is_vertical:
                layer_v_edges[from_layer] += 1

            # Check if edge follows layer direction rules
            if not self.geometry.is_valid_edge(from_x, from_y, from_layer, to_x, to_y, to_layer):
                illegal_edges += 1
                direction = self.geometry.layer_directions[from_layer]
                logger.error(f"ILLEGAL: {direction}-layer {from_layer} has wrong edge direction: {from_idx}->{to_idx}")

        if illegal_edges > 0:
            raise AssertionError(f"LATTICE FAIL: {illegal_edges} illegal edges found out of {total_edges_checked} checked")

        # Log H/V edge counts per layer and verify discipline
        for layer_id in range(self.geometry.layer_count):
            h_count = layer_h_edges[layer_id]
            v_count = layer_v_edges[layer_id]
            layer_dir = self.geometry.layer_directions[layer_id].upper()

            logger.info(f"[HV] L{layer_id} H_edges={h_count}, V_edges={v_count}")

            # ASSERTIONS per spec
            if layer_dir == "H":
                assert v_count == 0, f"[ASSERT] no vertical edges on H layer {layer_id} (found {v_count})"
            elif layer_dir == "V":
                assert h_count == 0, f"[ASSERT] no horizontal edges on V layer {layer_id} (found {h_count})"

        logger.info(f"LATTICE CORRECTNESS VERIFIED: {total_edges_checked} edges checked, all valid")


    def _verify_lattice_correctness(self, layers: int, x_steps: int, y_steps: int):
        """Build-time assertions for lattice correctness - verify no illegal edges exist"""
        logger.info("VERIFYING LATTICE CORRECTNESS...")
        
        # Build coordinate lookup for nodes
        node_coords = {}  # node_idx -> (x, y, layer, direction)
        layer_size = x_steps * y_steps
        
        for layer in range(layers):
            direction = 'h' if layer % 2 == 0 else 'v'  # H-layers: even, V-layers: odd
            
            for node_idx in range(layer * layer_size, (layer + 1) * layer_size):
                local_idx = node_idx - layer * layer_size
                x_idx = local_idx % x_steps
                y_idx = local_idx // x_steps
                node_coords[node_idx] = (x_idx, y_idx, layer, direction)
        
        # Count illegal edges
        horizontal_on_v_layers = 0
        vertical_on_h_layers = 0
        long_f_cu_edges = 0
        
        # Analyze all edges
        for from_idx, to_idx, cost in self.edges:
            if from_idx in node_coords and to_idx in node_coords:
                from_x, from_y, from_layer, from_dir = node_coords[from_idx]
                to_x, to_y, to_layer, to_dir = node_coords[to_idx]
                
                # Skip via connections (different layers)
                if from_layer != to_layer:
                    continue
                
                # Check edge direction vs layer direction
                is_horizontal_edge = (from_y == to_y and abs(from_x - to_x) == 1)
                is_vertical_edge = (from_x == to_x and abs(from_y - to_y) == 1)
                
                if is_horizontal_edge and from_dir == 'v':
                    horizontal_on_v_layers += 1
                    logger.error(f"ILLEGAL: H-edge on V-layer {from_layer}: {from_idx}->{to_idx}")
                
                if is_vertical_edge and from_dir == 'h':
                    vertical_on_h_layers += 1
                    logger.error(f"ILLEGAL: V-edge on H-layer {from_layer}: {from_idx}->{to_idx}")
                
                # Check F.Cu escape limit (layer 0)
                if from_layer == 0:
                    edge_length = abs(from_x - to_x) + abs(from_y - to_y)
                    if edge_length > 2:  # Max 2 grid steps
                        long_f_cu_edges += 1
                        logger.error(f"ILLEGAL: Long F.Cu edge length {edge_length}: {from_idx}->{to_idx}")
        
        # CRITICAL ASSERTIONS
        assert horizontal_on_v_layers == 0, f"LATTICE FAIL: {horizontal_on_v_layers} horizontal edges on V-layers"
        assert vertical_on_h_layers == 0, f"LATTICE FAIL: {vertical_on_h_layers} vertical edges on H-layers"  
        assert long_f_cu_edges == 0, f"LATTICE FAIL: {long_f_cu_edges} long F.Cu edges (>2 steps)"
        
        logger.info("LATTICE CORRECTNESS VERIFIED: No illegal edges found")
        
        # Unit spot checks: Pick 10 random nodes per layer type and verify neighbors
        self._spot_check_layer_neighbors(layers, layer_size, node_coords)
    

    def _spot_check_layer_neighbors(self, layers: int, layer_size: int, node_coords: dict):
        """Unit spot checks: verify neighbor connectivity follows layer rules"""
        import random
        
        for layer in range(layers):
            direction = 'h' if layer % 2 == 0 else 'v'
            layer_start = layer * layer_size
            layer_end = (layer + 1) * layer_size
            
            # Pick 10 random nodes on this layer
            sample_nodes = random.sample(range(layer_start, layer_end), min(10, layer_size))
            
            for node_idx in sample_nodes:
                neighbors = self._get_node_neighbors(node_idx)
                node_x, node_y, node_layer, node_dir = node_coords[node_idx]
                
                for neighbor_idx in neighbors:
                    if neighbor_idx in node_coords:
                        neigh_x, neigh_y, neigh_layer, neigh_dir = node_coords[neighbor_idx]
                        
                        # Skip vias (different layers)
                        if node_layer != neigh_layer:
                            continue
                        
                        if direction == 'h':
                            # H-layer: neighbors should only differ in X
                            assert neigh_y == node_y, f"H-layer neighbor differs in Y: {node_idx}->{neighbor_idx}"
                            assert abs(neigh_x - node_x) == 1, f"H-layer neighbor not adjacent in X: {node_idx}->{neighbor_idx}"
                        else:
                            # V-layer: neighbors should only differ in Y  
                            assert neigh_x == node_x, f"V-layer neighbor differs in X: {node_idx}->{neighbor_idx}"
                            assert abs(neigh_y - node_y) == 1, f"V-layer neighbor not adjacent in Y: {node_idx}->{neighbor_idx}"
            
            logger.info(f"Layer {layer} ({direction}): {len(sample_nodes)} nodes verified")
    

    def _get_node_neighbors(self, node_idx: int) -> List[int]:
        """Get all neighbors of a node from edge list"""
        neighbors = []
        for from_idx, to_idx, cost in self.edges:
            if from_idx == node_idx:
                neighbors.append(to_idx)
        return neighbors
    

    def _connect_pads_optimized(self, pads: List[Pad]):
        """ESCAPE ROUTING: Generate escape stubs with vias aligned to routing grid"""
        logger.info(f"Connecting {len(pads)} pads with escape routing strategy...")

        # PERFORMANCE MONITORING: Track escape phase timing and progress
        import time
        start_time = time.time()
        connected = 0
        blocked_escapes = 0

        for k, pad in enumerate(pads):
            # Progress logging every 400 pads (throttled for Windows performance)
            if (k + 1) % 400 == 0:
                elapsed = time.time() - start_time
                logger.info(f"[ESCAPE] progress={k+1}/{len(pads)} elapsed={elapsed:.1f}s connected={connected} blocked={blocked_escapes}")

            # Original pad processing logic follows
            try:
                # Get compatible pad attributes
                net_name = self._get_pad_net_name(pad)
                x_mm, y_mm = self._get_pad_coordinates(pad)

                # 1. Create pad node - CRITICAL FIX: Add to coordinate arrays
                pad_node_id = f"pad_{net_name}_{x_mm:.1f}_{y_mm:.1f}"
                self.nodes[pad_node_id] = (x_mm, y_mm, 0, self.node_count)
                self._node_lookup[pad_node_id] = self.node_count
                pad_idx = self.node_count

                # CRITICAL FIX: Add node to coordinate arrays that spatial indexing uses
                pad_coords = [x_mm, y_mm, 0.0]
                
                # EFFICIENT BATCH EXTENSION: Store pad coordinates for later batch update
                if not hasattr(self, '_pending_coordinate_extensions'):
                    self._pending_coordinate_extensions = []
                self._pending_coordinate_extensions.append(pad_coords)
                    
                self.node_count += 1
                
                # 2. Generate escape stub (5mm vertical outward from board interior)
                escape_success = self._create_escape_stub(pad, pad_idx)
                
                if escape_success:
                    connected += 1
                else:
                    blocked_escapes += 1
                    logger.warning(f"Pad escape blocked for {net_name} at ({x_mm:.1f}, {y_mm:.1f})")

            except Exception as e:
                net_name = self._get_pad_net_name(pad)
                logger.error(f"Failed to connect pad {net_name}: {e}")
                blocked_escapes += 1
        
        # BATCH COORDINATE EXTENSION: Apply all pending coordinate extensions at once
        if hasattr(self, '_pending_coordinate_extensions') and len(self._pending_coordinate_extensions) > 0:
            logger.info(f"BATCH COORD EXTENSION: Processing {len(self._pending_coordinate_extensions)} escape node coordinates...")
            
            if self.node_coordinates is not None:
                # Create batch of new coordinates
                new_coords_array = np.array(self._pending_coordinate_extensions)
                
                if self.use_gpu:
                    # GPU batch extension
                    existing_coords = self.node_coordinates.get() if hasattr(self.node_coordinates, 'get') else self.node_coordinates
                    batch_coords_gpu = cp.array(new_coords_array)
                    self.node_coordinates = cp.vstack([cp.array(existing_coords), batch_coords_gpu])
                    logger.info(f"BATCH COORD: GPU extended from {existing_coords.shape[0]} to {self.node_coordinates.shape[0]} rows")
                else:
                    # CPU batch extension
                    old_count = self.node_coordinates.shape[0]
                    self.node_coordinates = np.vstack([self.node_coordinates, new_coords_array])
                    logger.info(f"BATCH COORD: CPU extended from {old_count} to {self.node_coordinates.shape[0]} rows")
                
                # Clear the pending list
                self._pending_coordinate_extensions.clear()

                # CRITICAL FIX: Update node count consistency after coordinate extension
                if hasattr(self, 'node_coordinates_lattice'):
                    # Ensure we're using the extended coordinates as the lattice
                    self.node_coordinates_lattice = self.node_coordinates

                # Set N on both self and graph_state from the final coordinate array
                self.lattice_node_count = int(self.node_coordinates.shape[0])
                if getattr(self, "graph_state", None) is not None:
                    self.graph_state.lattice_node_count = self.lattice_node_count

                logger.info(f"[NODE-COUNT-SYNC] Updated lattice_node_count to {self.lattice_node_count} after coordinate extension")
            else:
                logger.error("BATCH COORD BUG: node_coordinates is None - cannot perform batch extension!")
        
        # PERFORMANCE MONITORING: Final escape phase summary
        total_elapsed = time.time() - start_time
        logger.info(f"[ESCAPE] complete: {connected}/{len(pads)} connected, {blocked_escapes} blocked, elapsed={total_elapsed:.2f}s")
    

    def _create_escape_stub(self, pad, pad_idx: int) -> bool:
        """Create escape stub with via aligned to routing grid"""

        # Get compatible pad attributes
        net_name = self._get_pad_net_name(pad)
        x_mm, y_mm = self._get_pad_coordinates(pad)

        # Use stored board bounds for escape direction
        bounds = self._board_bounds
        board_center_x = (bounds.min_x + bounds.max_x) / 2
        board_center_y = (bounds.min_y + bounds.max_y) / 2

        # Implement proper stub-then-via emission with strict pad clearance
        net_width_mm = getattr(pad, 'width_mm', 0.2)  # Track width
        net_clearance_mm = getattr(pad, 'clearance_mm', PAD_CLEARANCE_MM)  # DRC clearance

        # CONFIG constants
        MIN_STUB_MM = max(net_width_mm * 2.0, 0.25)  # Visible stub length
        PAD_CLEAR_MM = max(net_clearance_mm, PAD_CLEARANCE_MM)   # Spacing from pad edge
        GRID = self.config.grid_pitch

        # Get pad attributes
        pad_xy = (x_mm, y_mm)
        pad_layer = self._get_pad_layer(pad)  # F.Cu, B.Cu, etc.

        # Calculate escape direction: away from board center for better distribution
        escape_dx = 1 if x_mm >= board_center_x else -1
        escape_dy = 1 if y_mm >= board_center_y else -1

        # Initial portal position with minimum stub length
        portal_x = x_mm + escape_dx * MIN_STUB_MM
        portal_y = y_mm + escape_dy * MIN_STUB_MM

        # Snap to routing grid
        portal_x = round(portal_x / GRID) * GRID
        portal_y = round(portal_y / GRID) * GRID
        portal_xy = (portal_x, portal_y)

        # Vector from pad to portal
        v = (portal_xy[0] - pad_xy[0], portal_xy[1] - pad_xy[1])
        if (v[0]**2 + v[1]**2)**0.5 < MIN_STUB_MM:
            # Push along nearest axis so stub is not zero and via is outside pad
            if abs(v[0]) >= abs(v[1]):
                dv = (GRID * escape_dx, 0.0)
            else:
                dv = (0.0, GRID * escape_dy)
            portal_xy = (pad_xy[0] + dv[0], pad_xy[1] + dv[1])

        # Ensure via landing is outside pad clearance
        # Simulate pad.distance_to_edge() - for now use simple radius check
        pad_width = getattr(pad, 'width', getattr(pad, 'size_x', 1.0))
        pad_height = getattr(pad, 'height', getattr(pad, 'size_y', 1.0))
        pad_radius = max(pad_width, pad_height) * 0.5
        distance_to_pad_center = ((portal_xy[0] - pad_xy[0])**2 + (portal_xy[1] - pad_xy[1])**2)**0.5

        if distance_to_pad_center < pad_radius + PAD_CLEAR_MM:
            # Move portal further out to meet clearance
            required_distance = pad_radius + PAD_CLEAR_MM
            scale = required_distance / max(distance_to_pad_center, 1e-6)
            portal_xy = (pad_xy[0] + (portal_xy[0] - pad_xy[0]) * scale,
                        pad_xy[1] + (portal_xy[1] - pad_xy[1]) * scale)
            # Re-snap to grid
            portal_xy = (round(portal_xy[0] / GRID) * GRID, round(portal_xy[1] / GRID) * GRID)

        grid_x, grid_y = portal_xy

        # 2. Create stub end node (where stub connects to routing lattice)
        stub_end_id = f"stub_{net_name}_{grid_x:.1f}_{grid_y:.1f}"
        self.nodes[stub_end_id] = (grid_x, grid_y, pad_layer, self.node_count)
        self._node_lookup[stub_end_id] = self.node_count
        stub_end_idx = self.node_count

        # Cache coordinates for later addition to coordinate array
        stub_coords = [grid_x, grid_y, float(pad_layer)]
        if not hasattr(self, '_pending_coordinate_extensions'):
            self._pending_coordinate_extensions = []
        self._pending_coordinate_extensions.append(stub_coords)
        self.node_count += 1

        # 4. Create escape via node on routing layer (prefer inner layers)
        via_layer = 1 if pad_layer == 0 else 2 if self.layer_count > 2 else 0  # Use inner layers
        via_node_id = f"via_{net_name}_{grid_x:.1f}_{grid_y:.1f}"
        self.nodes[via_node_id] = (grid_x, grid_y, via_layer, self.node_count)
        self._node_lookup[via_node_id] = self.node_count
        via_idx = self.node_count

        # Add via node to coordinate arrays
        via_coords = [grid_x, grid_y, float(via_layer)]
        self._pending_coordinate_extensions.append(via_coords)
        self.node_count += 1

        # 5. Connect pad to stub end (stub on pad layer)
        stub_cost = HISTORICAL_ACCUMULATION  # Low cost for pad connection stub
        self.edges.extend([(pad_idx, stub_end_idx, stub_cost), (stub_end_idx, pad_idx, stub_cost)])

        # 6. Connect stub end to via (layer transition)
        via_cost = 0.2 * self.geometry.pitch  # Small positive cost to prevent via stacking
        self.edges.extend([(stub_end_idx, via_idx, via_cost), (via_idx, stub_end_idx, via_cost)])
        
        # 5. Connect via into routing lattice
        lattice_connected = self._connect_via_to_lattice(via_idx, grid_x, grid_y)
        
        if lattice_connected:
            logger.debug(f"Escape created: {net_name} → via at ({grid_x:.1f}, {grid_y:.1f})")
            return True
        else:
            logger.warning(f"Via {via_node_id} could not connect to lattice")
            return False
    

    def _connect_via_to_lattice(self, via_idx: int, grid_x: float, grid_y: float) -> bool:
        """Connect escape via to routing lattice at grid coordinates"""
        
        # Find lattice nodes at this grid position on multiple layers
        connected_layers = 0
        via_cost = 0.2 * self.geometry.pitch  # Small positive cost to prevent via stacking
        
        # Connect to layer 0 (H-layer) and layer 1 (V-layer) if they exist
        for layer in range(self.geometry.layer_count):
            if layer in self._spatial_index:
                # Find exact grid coordinate match in this layer  
                closest_node = None
                min_distance = float('inf')
                
                for rail_x, rail_y, rail_node_id, rail_idx in self._spatial_index[layer]:
                    distance = ((rail_x - grid_x)**2 + (rail_y - grid_y)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = (rail_x, rail_y, rail_node_id, rail_idx)
                    
                    # Check for exact grid alignment (within small tolerance)
                    if abs(rail_x - grid_x) < 0.01 and abs(rail_y - grid_y) < 0.01:
                        # Connect via to this lattice node
                        self.edges.extend([(via_idx, rail_idx, via_cost), (rail_idx, via_idx, via_cost)])
                        connected_layers += 1
                        logger.debug(f"Via connected to layer {layer} node {rail_node_id} at exact match")
                        break
                
                # If no exact match, log the closest node for debugging
                if connected_layers == 0 and closest_node:
                    rail_x, rail_y, rail_node_id, rail_idx = closest_node
                    logger.debug(f"Via at ({grid_x:.1f}, {grid_y:.1f}) - closest layer {layer} node: {rail_node_id} at ({rail_x:.1f}, {rail_y:.1f}), distance: {min_distance:.3f}")
                    
                    # If very close (within 1.0mm), connect anyway
                    if min_distance < 1.0:
                        self.edges.extend([(via_idx, rail_idx, via_cost), (rail_idx, via_idx, via_cost)])
                        connected_layers += 1
                        logger.debug(f"Via connected to layer {layer} node {rail_node_id} (close match, distance: {min_distance:.3f})")
        
        return connected_layers > 0
    

    def _find_local_rails_at_position(self, x: float, y: float) -> List[str]:
        """Find rails at this X,Y position on multiple layers to prevent bottlenecks"""
        local_rails = []
        
        # Connect to rails on layer 0 (F.Cu) and layer 1 (first routing layer)
        for layer in range(self.geometry.layer_count):
            if layer in self._spatial_index:
                layer_nodes = self._spatial_index[layer]
                
                # Find rails within small distance of this pad position
                for rail_x, rail_y, node_id, idx in layer_nodes:
                    # Look for rails at this approximate X position (within 1 grid pitch)
                    if abs(rail_x - x) <= self.config.grid_pitch and abs(rail_y - y) <= 2.0:
                        local_rails.append(node_id)
                        break  # Only need one rail per layer
        
        return local_rails
    

    def _find_nearest_rail_fast(self, x: float, y: float, layer: int, max_dist: float) -> Optional[str]:
        """O(1) spatial lookup for nearest rail"""
        if layer not in self._spatial_index:
            return None
        
        layer_nodes = self._spatial_index[layer]
        best_rail = None
        min_dist = max_dist
        
        # Linear search within layer (small constant factor since layer nodes are spatially organized)
        for rail_x, rail_y, node_id, idx in layer_nodes:
            if abs(rail_x - x) > max_dist or abs(rail_y - y) > max_dist:
                continue  # Quick bounding box check
            
            dist = ((x - rail_x)**2 + (y - rail_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                best_rail = node_id
        
        return best_rail


    def _refresh_edge_dependent_arrays(self):
        """Resize/recreate arrays that must match the directed-edge count."""
        gs = getattr(self, "graph_state", None)

        # Prefer graph_state; fall back to attributes bound on self
        indices = getattr(gs, "indices_cpu", getattr(self, "indices_cpu", None))
        if indices is None:
            return

        import numpy as np
        edge_count = int(len(indices))
        self.edge_count = edge_count
        if gs is not None:
            gs.edge_count = edge_count

        if self.use_gpu:
            import cupy as cp

        def ensure(name, dtype, fill=0):
            arr = getattr(self, name, None)
            if arr is None or len(arr) != edge_count or arr.dtype != dtype:
                if self.use_gpu:
                    new = cp.full(edge_count, fill, dtype=dtype)
                else:
                    new = np.full(edge_count, fill, dtype=dtype)
                setattr(self, name, new)

        # Keep all as 1-D arrays matching edge count
        ensure("edge_total_penalty", np.float32 if not self.use_gpu else cp.float32, 0.0)
        ensure("edge_dir_mask", np.uint8   if not self.use_gpu else cp.uint8,   1)
        ensure("edge_bottleneck_penalty", np.float32 if not self.use_gpu else cp.float32, 0.0)

        # Also refresh edge state arrays
        ensure("edge_present_usage", np.float32 if not self.use_gpu else cp.float32, 0.0)
        ensure("edge_history", np.float32 if not self.use_gpu else cp.float32, 0.0)
        ensure("edge_capacity", np.float32 if not self.use_gpu else cp.float32, 1.0)
        ensure("edge_total_cost", np.float32 if not self.use_gpu else cp.float32, 0.0)

        # Update legacy aliases for compatibility
        self.congestion = self.edge_present_usage
        self.history_cost = self.edge_history
        self.edge_mask = self.edge_dir_mask  # back-compat for any lingering callers

        logger.info("[EDGE-REFRESH] edge_count=%d (penalty=%d dir=%d bottle=%d usage=%d)",
                    edge_count,
                    len(self.edge_total_penalty),
                    len(self.edge_dir_mask),
                    len(self.edge_bottleneck_penalty),
                    len(self.edge_present_usage))


    def _refresh_edge_arrays_after_portal_bind(self):
        """Re-sync all edge-length–dependent arrays after inserting portal edges."""
        gs = getattr(self, "graph_state", self)
        # E_live is the authoritative edge count in the live CSR
        indices = getattr(gs, "indices_cpu", getattr(self, "indices_cpu", None))
        if indices is None:
            raise RuntimeError("[LIVE-SIZE] indices_cpu not available in _refresh_edge_arrays_after_portal_bind")
        E_live = len(indices)
        self.E = E_live

        # Use centralized live-size contract helper
        self.on_live_size_changed(E_live)


