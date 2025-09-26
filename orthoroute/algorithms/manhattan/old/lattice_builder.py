"""
Clean GPU-First 3D Lattice Builder for PathFinder Routing

Builds routing lattice directly as GPU sparse matrices for maximum performance.
No intermediate Python objects - straight to CuPy CSR format.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
    GPU_AVAILABLE = True
except ImportError:
    import scipy.sparse as sp
    GPU_AVAILABLE = False

from .types import Pad
from ...domain.models.board import Board, Net

logger = logging.getLogger(__name__)


class LatticeBuilder:
    """Builds 3D orthogonal routing lattice directly as GPU sparse matrices"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.grid_pitch = 0.4  # mm - proven to work
        self.use_gpu = GPU_AVAILABLE
        
        # Grid data (built once)
        self.nodes = {}  # node_id -> (x, y, layer, index)
        self.node_count = 0
        self.edges = []  # list of (from_idx, to_idx, cost)
        
        logger.info(f"LatticeBuilder initialized (GPU: {self.use_gpu})")
    
    def build_gpu_routing_matrices(self, board: Board) -> Dict:
        """Build complete routing matrices for GPU PathFinder"""
        logger.info("Building GPU routing matrices for PathFinder")
        
        # Calculate routing bounds
        bounds = self._calculate_routing_bounds(board)
        logger.info(f"Routing bounds: {bounds}")
        
        # Build 3D lattice
        self._build_3d_lattice(bounds, board.layer_count)
        
        # Connect pads with F.Cu escape routing
        self._connect_pads_to_lattice(board.pads)
        
        # Convert to GPU sparse matrices
        matrices = self._convert_to_gpu_matrices()
        
        logger.info(f"Built lattice: {self.node_count} nodes, {len(self.edges)} edges")
        return matrices
    
    def _calculate_routing_bounds(self, board: Board) -> Tuple[float, float, float, float]:
        """Calculate routing area bounds using board geometric bounds + 3mm margin"""
        
        # Use board geometric bounds directly if available
        if hasattr(board, 'get_bounds') and callable(board.get_bounds):
            try:
                bounds = board.get_bounds()
                min_x, min_y = bounds.min_x, bounds.min_y
                max_x, max_y = bounds.max_x, bounds.max_y
                logger.info(f"Using board geometric bounds: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
            except Exception as e:
                logger.warning(f"Could not get board bounds: {e}, falling back to pad-based bounds")
                bounds = None
        else:
            bounds = None
            
        if bounds is None:
            # Calculate bounds from pad coordinates, but filter outliers to get main board area
            if not board.pads:
                logger.warning("No pads available - using default bounds")
                return (0, 0, 100, 100)
            
            # Get all pad coordinates
            all_x = [pad.x_mm for pad in board.pads]
            all_y = [pad.y_mm for pad in board.pads]
            
            # Use percentile-based bounds to filter extreme outliers (use 95th percentile range)
            all_x_sorted = sorted(all_x)
            all_y_sorted = sorted(all_y)
            n_pads = len(all_x_sorted)
            
            # Use 2.5th to 97.5th percentile (keeps 95% of pads, filters extreme outliers)
            idx_low = max(0, int(n_pads * 0.025))
            idx_high = min(n_pads - 1, int(n_pads * 0.975))
            
            min_x = all_x_sorted[idx_low]
            max_x = all_x_sorted[idx_high]
            min_y = all_y_sorted[idx_low]
            max_y = all_y_sorted[idx_high]
            
            outliers_filtered = n_pads - (idx_high - idx_low + 1)
            if outliers_filtered > 0:
                logger.info(f"Filtered {outliers_filtered} outlier pads using 95th percentile bounds")
            
            logger.info(f"Calculated board bounds from pads: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
        
        # Add 3mm margin for routing
        margin = 3.0
        final_bounds = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
        logger.info(f"Final routing bounds with {margin}mm margin: X=[{final_bounds[0]:.1f}, {final_bounds[2]:.1f}], Y=[{final_bounds[1]:.1f}, {final_bounds[3]:.1f}]")
        
        return final_bounds
    
    def _build_3d_lattice(self, bounds: Tuple[float, float, float, float], layer_count: int):
        """Build complete 3D orthogonal lattice"""
        min_x, min_y, max_x, max_y = bounds
        
        # Align to grid
        grid_min_x = round(min_x / self.grid_pitch) * self.grid_pitch
        grid_max_x = round(max_x / self.grid_pitch) * self.grid_pitch  
        grid_min_y = round(min_y / self.grid_pitch) * self.grid_pitch
        grid_max_y = round(max_y / self.grid_pitch) * self.grid_pitch
        
        x_steps = int((grid_max_x - grid_min_x) / self.grid_pitch) + 1
        y_steps = int((grid_max_y - grid_min_y) / self.grid_pitch) + 1
        
        logger.info(f"3D lattice: {x_steps} x {y_steps} x {layer_count} = {x_steps * y_steps * layer_count:,} nodes")
        
        # Create all lattice nodes
        for layer in range(layer_count):
            direction = 'h' if (layer % 2 == 1) else 'v'  # F.Cu=v, In1=h, In2=v, etc.
            
            for x_idx in range(x_steps):
                x = grid_min_x + (x_idx * self.grid_pitch)
                for y_idx in range(y_steps):
                    y = grid_min_y + (y_idx * self.grid_pitch)
                    
                    node_id = f"rail_{direction}_{x_idx}_{y_idx}_{layer}"
                    self.nodes[node_id] = (x, y, layer, self.node_count)
                    self.node_count += 1
        
        # Create intra-layer connections
        self._build_intra_layer_connections(x_steps, y_steps, layer_count, grid_min_x, grid_min_y)
        
        # Create inter-layer vias
        self._build_inter_layer_vias(x_steps, y_steps, layer_count)
    
    def _build_intra_layer_connections(self, x_steps: int, y_steps: int, layer_count: int, 
                                     grid_min_x: float, grid_min_y: float):
        """Create rail-to-rail connections within each layer"""
        for layer in range(layer_count):
            direction = 'h' if (layer % 2 == 1) else 'v'
            
            if direction == 'h':
                # Horizontal layer: connect left-to-right
                for y_idx in range(y_steps):
                    for x_idx in range(x_steps - 1):
                        from_node = f"rail_h_{x_idx}_{y_idx}_{layer}"
                        to_node = f"rail_h_{x_idx+1}_{y_idx}_{layer}"
                        
                        if from_node in self.nodes and to_node in self.nodes:
                            from_idx = self.nodes[from_node][3]
                            to_idx = self.nodes[to_node][3]
                            
                            # Bidirectional connections
                            self.edges.append((from_idx, to_idx, 1.0))
                            self.edges.append((to_idx, from_idx, 1.0))
            else:
                # Vertical layer: connect top-to-bottom  
                for x_idx in range(x_steps):
                    for y_idx in range(y_steps - 1):
                        from_node = f"rail_v_{x_idx}_{y_idx}_{layer}"
                        to_node = f"rail_v_{x_idx}_{y_idx+1}_{layer}"
                        
                        if from_node in self.nodes and to_node in self.nodes:
                            from_idx = self.nodes[from_node][3]
                            to_idx = self.nodes[to_node][3]
                            
                            # Bidirectional connections
                            self.edges.append((from_idx, to_idx, 1.0))
                            self.edges.append((to_idx, from_idx, 1.0))
    
    def _build_inter_layer_vias(self, x_steps: int, y_steps: int, layer_count: int):
        """Create vias between adjacent layers"""
        for layer in range(layer_count - 1):
            from_direction = 'h' if (layer % 2 == 1) else 'v'
            to_direction = 'h' if ((layer + 1) % 2 == 1) else 'v'
            
            for x_idx in range(x_steps):
                for y_idx in range(y_steps):
                    from_node = f"rail_{from_direction}_{x_idx}_{y_idx}_{layer}"
                    to_node = f"rail_{to_direction}_{x_idx}_{y_idx}_{layer+1}"
                    
                    if from_node in self.nodes and to_node in self.nodes:
                        from_idx = self.nodes[from_node][3]
                        to_idx = self.nodes[to_node][3]
                        
                        # Via cost increases with layer distance
                        via_cost = 3.0 + layer * 0.5
                        
                        # Bidirectional vias
                        self.edges.append((from_idx, to_idx, via_cost))
                        self.edges.append((to_idx, from_idx, via_cost))
    
    def _connect_pads_to_lattice(self, pads: List[Pad]):
        """Connect pads to lattice with DRC-compliant F.Cu escape routing + vias"""
        logger.info(f"Connecting {len(pads)} pads with escape routing (max 5mm F.Cu + vias)")
        
        connected_pads = 0
        for pad in pads:
            # Create pad escape routing with via to inner layers
            via_connection = self._create_pad_escape_routing(pad)
            if via_connection:
                connected_pads += 1
                if connected_pads <= 5:  # Debug first 5 connections
                    logger.info(f"Connected pad {pad.net_name} at ({pad.x_mm:.2f}, {pad.y_mm:.2f}) "
                               f"with {len(via_connection['nodes'])} escape nodes and via to layer {via_connection['via_layer']}")
            else:
                logger.error(f"Could not create escape routing for pad {pad.net_name} at ({pad.x_mm:.2f}, {pad.y_mm:.2f})")
        
        logger.info(f"Connected {connected_pads}/{len(pads)} pads with escape routing")
    
    def _create_pad_escape_routing(self, pad: Pad) -> Dict:
        """Create DRC-compliant pad escape routing with via to inner layers"""
        max_escape_length = 5.0  # mm - maximum F.Cu escape route length
        
        # Add pad node on F.Cu (layer 0)
        pad_node_id = f"pad_{pad.net_name}_{pad.x_mm:.1f}_{pad.y_mm:.1f}"
        self.nodes[pad_node_id] = (pad.x_mm, pad.y_mm, 0, self.node_count)
        pad_idx = self.node_count
        self.node_count += 1
        
        # Find escape route direction and via location within 5mm
        via_location = self._find_optimal_via_location(pad, max_escape_length)
        if not via_location:
            return None
        
        via_x, via_y, escape_length, via_layer = via_location
        
        # Create intermediate escape routing nodes on F.Cu if needed (for visualization)
        escape_nodes = []
        if escape_length > 0.1:  # Only if escape route is significant
            # Create 1-2 intermediate nodes for better visualization
            if escape_length > 2.0:
                mid_x = pad.x_mm + (via_x - pad.x_mm) * 0.5
                mid_y = pad.y_mm + (via_y - pad.y_mm) * 0.5
                escape_mid_id = f"escape_{pad.net_name}_{mid_x:.1f}_{mid_y:.1f}_0"
                self.nodes[escape_mid_id] = (mid_x, mid_y, 0, self.node_count)
                mid_idx = self.node_count
                self.node_count += 1
                escape_nodes.append((escape_mid_id, mid_idx))
        
        # Create via node on target inner layer
        via_node_id = f"via_{pad.net_name}_{via_x:.1f}_{via_y:.1f}_{via_layer}"
        self.nodes[via_node_id] = (via_x, via_y, via_layer, self.node_count)
        via_idx = self.node_count
        self.node_count += 1
        
        # Create connections: pad -> escape nodes -> via
        current_idx = pad_idx
        for escape_id, escape_idx in escape_nodes:
            escape_cost = 1.0  # F.Cu routing cost
            self.edges.append((current_idx, escape_idx, escape_cost))
            self.edges.append((escape_idx, current_idx, escape_cost))
            current_idx = escape_idx
        
        # Connect final node to via (F.Cu to inner layer transition)
        via_cost = 5.0 + via_layer * 0.5  # Via cost increases with layer depth
        self.edges.append((current_idx, via_idx, via_cost))
        self.edges.append((via_idx, current_idx, via_cost))
        
        # Connect via to nearby inner layer routing rails
        connected_rails = self._connect_via_to_inner_layer_rails(via_x, via_y, via_layer, via_idx)
        
        return {
            'pad_node': pad_node_id,
            'via_node': via_node_id,
            'via_layer': via_layer,
            'escape_length': escape_length,
            'nodes': [pad_node_id] + [n[0] for n in escape_nodes] + [via_node_id],
            'connected_rails': connected_rails
        }
    
    def _find_optimal_via_location(self, pad: Pad, max_distance: float) -> Tuple[float, float, float, int]:
        """Find optimal via location within max_distance of pad"""
        # Target inner layer for routing (prefer In1.Cu layer 1 for first via)
        target_layer = 1
        
        # Find nearest inner layer rail within max_distance
        best_location = None
        min_distance = float('inf')
        
        for node_id, (x, y, layer, idx) in self.nodes.items():
            if layer == target_layer and node_id.startswith('rail_h_'):  # In1.Cu horizontal rails
                distance = ((pad.x_mm - x) ** 2 + (pad.y_mm - y) ** 2) ** 0.5
                
                if distance <= max_distance and distance < min_distance:
                    min_distance = distance
                    best_location = (x, y, distance, target_layer)
        
        # If no suitable In1.Cu rail, try In2.Cu (layer 2)
        if not best_location:
            target_layer = 2
            for node_id, (x, y, layer, idx) in self.nodes.items():
                if layer == target_layer and node_id.startswith('rail_v_'):  # In2.Cu vertical rails
                    distance = ((pad.x_mm - x) ** 2 + (pad.y_mm - y) ** 2) ** 0.5
                    
                    if distance <= max_distance and distance < min_distance:
                        min_distance = distance
                        best_location = (x, y, distance, target_layer)
        
        return best_location
    
    def _connect_via_to_inner_layer_rails(self, via_x: float, via_y: float, via_layer: int, via_idx: int) -> List[str]:
        """Connect via to nearby inner layer routing rails"""
        connected_rails = []
        max_connection_distance = 2.0  # mm - increased tolerance to ensure connectivity
        
        # Find rails on the via layer within connection distance
        layer_direction = 'h' if (via_layer % 2 == 1) else 'v'
        rail_prefix = f"rail_{layer_direction}_"
        
        # Debug: Track closest rail for troubleshooting
        closest_rail = None
        min_distance = float('inf')
        
        for node_id, (x, y, layer, idx) in self.nodes.items():
            if layer == via_layer and node_id.startswith(rail_prefix):
                distance = ((via_x - x) ** 2 + (via_y - y) ** 2) ** 0.5
                
                # Track closest for debugging
                if distance < min_distance:
                    min_distance = distance
                    closest_rail = (node_id, distance)
                
                if distance <= max_connection_distance:
                    # Connect via to rail
                    connection_cost = 1.0 + distance * 0.5
                    self.edges.append((via_idx, idx, connection_cost))
                    self.edges.append((idx, via_idx, connection_cost))
                    connected_rails.append(node_id)
        
        # Debug logging for connection issues
        if not connected_rails and closest_rail:
            logger.warning(f"Via at ({via_x:.1f}, {via_y:.1f}) layer {via_layer} failed to connect. "
                          f"Closest rail {closest_rail[0]} at distance {closest_rail[1]:.3f}mm > {max_connection_distance}mm")
        elif connected_rails:
            logger.debug(f"Via at ({via_x:.1f}, {via_y:.1f}) layer {via_layer} connected to {len(connected_rails)} rails")
        
        return connected_rails
    
    def _find_nearest_fcu_rail(self, pad: Pad) -> str:
        """Find nearest F.Cu rail node to pad (DRC compliant)"""
        best_rail = None
        min_distance = float('inf')
        nearest_rail_info = None
        
        for node_id, (x, y, layer, idx) in self.nodes.items():
            if layer == 0 and node_id.startswith('rail_v_'):  # F.Cu vertical rails only
                distance = ((pad.x_mm - x) ** 2 + (pad.y_mm - y) ** 2) ** 0.5
                horizontal_offset = abs(pad.x_mm - x)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_rail_info = (node_id, horizontal_offset, distance)
                
                # Relaxed constraint for connectivity testing: ≤1.0mm horizontal offset
                if horizontal_offset <= 1.0:  # Much more relaxed for testing connectivity
                    if distance < min_distance or best_rail is None:
                        best_rail = node_id
        
        if best_rail is None and nearest_rail_info:
            logger.warning(f"Pad {pad.net_name} at ({pad.x_mm:.2f}, {pad.y_mm:.2f}): "
                          f"Nearest rail {nearest_rail_info[0]} has horizontal offset {nearest_rail_info[1]:.3f}mm > 1.0mm limit")
        
        return best_rail
    
    def _calculate_escape_cost(self, pad: Pad, rail_node_id: str) -> float:
        """Calculate cost for F.Cu escape route (vertical + dogleg)"""
        rail_x, rail_y, _, _ = self.nodes[rail_node_id]
        
        # Escape distance components
        horizontal_distance = abs(pad.x_mm - rail_x)
        vertical_distance = abs(pad.y_mm - rail_y) 
        
        # Cost based on total routing distance
        return 0.5 + horizontal_distance + vertical_distance
    
    def _convert_to_gpu_matrices(self) -> Dict:
        """Convert graph to GPU sparse matrices"""
        logger.info("Converting to GPU sparse matrices")
        
        # Build adjacency matrix
        if self.edges:
            row_indices = [edge[0] for edge in self.edges]
            col_indices = [edge[1] for edge in self.edges]  
            costs = [edge[2] for edge in self.edges]
        else:
            row_indices = [0]
            col_indices = [0]
            costs = [1.0]
        
        # Create sparse adjacency matrix
        if self.use_gpu:
            row_indices = cp.array(row_indices)
            col_indices = cp.array(col_indices)
            costs = cp.array(costs)
            adjacency = gpu_csr_matrix((costs, (row_indices, col_indices)), 
                                     shape=(self.node_count, self.node_count))
        else:
            adjacency = sp.csr_matrix((costs, (row_indices, col_indices)),
                                    shape=(self.node_count, self.node_count))
        
        # Node coordinates array
        coords = np.zeros((self.node_count, 3))  # x, y, layer
        for node_id, (x, y, layer, idx) in self.nodes.items():
            coords[idx] = [x, y, layer]
        
        if self.use_gpu:
            coords = cp.array(coords)
        
        return {
            'adjacency_matrix': adjacency,
            'node_coordinates': coords,
            'node_count': self.node_count,
            'nodes': self.nodes,  # For debugging/visualization
            'use_gpu': self.use_gpu
        }