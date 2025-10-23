"""
Routing Resource Graph (RRG) for FPGA-style Manhattan routing
Based on PathFinder negotiated congestion algorithm
"""

import heapq
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of RRG nodes"""
    RAIL = "rail"           # Vertical track segment
    BUS = "bus"             # Horizontal track segment  
    SWITCH = "switch"       # Rail/bus intersection for layer changes
    PAD_ENTRY = "pad_entry" # F.Cu pad entry point
    PAD_EXIT = "pad_exit"   # F.Cu pad exit point

class EdgeType(Enum):
    """Types of RRG edges"""
    TRACK = "track"         # Along same track/layer
    SWITCH = "switch"       # Between layers at switch box
    ENTRY = "entry"         # F.Cu pad to fabric
    EXIT = "exit"           # Fabric to F.Cu pad

@dataclass
class RRGNode:
    """RRG node representation"""
    id: str
    node_type: NodeType
    x: float                # World coordinates
    y: float
    layer: int             # Layer index (0=In1.Cu, 1=In2.Cu, etc.)
    capacity: int = 1      # Track capacity
    usage: int = 0         # Current usage
    track_index: int = 0   # Which track/lane on this layer
    
    def is_available(self) -> bool:
        """Check if node has available capacity"""
        return self.usage < self.capacity
    
    def utilization(self) -> float:
        """Get utilization ratio"""
        return self.usage / self.capacity if self.capacity > 0 else 0.0

@dataclass 
class RRGEdge:
    """RRG edge representation"""
    id: str
    edge_type: EdgeType
    from_node: str         # Node IDs
    to_node: str
    length_mm: float       # Physical length
    capacity: int = 1      # Edge capacity
    usage: int = 0         # Current usage
    base_cost: float = 0.0 # Base routing cost
    history_cost: float = 0.0  # Accumulated congestion penalty
    
    def is_available(self) -> bool:
        """Check if edge has available capacity"""
        return self.usage < self.capacity
    
    def utilization(self) -> float:
        """Get utilization ratio"""
        return self.usage / self.capacity if self.capacity > 0 else 0.0
    
    def current_cost(self, pres_fac: float = 1.0, alpha: float = 2.0) -> float:
        """Calculate current routing cost with congestion"""
        present_penalty = pres_fac * (self.utilization() ** alpha) if self.usage > 0 else 0.0
        return self.base_cost * (1.0 + present_penalty) + self.history_cost

@dataclass
class RoutingConfig:
    """PathFinder routing configuration"""
    grid_pitch: float = 0.4        # mm between tracks
    track_width: float = 0.0889    # mm trace width
    clearance: float = 0.0889      # mm minimum spacing
    via_diameter: float = 0.25     # mm
    via_drill: float = 0.15        # mm
    
    # Cost weights
    k_length: float = 1.0          # Length cost weight
    k_via: float = 10.0            # Via cost weight  
    k_bend: float = 2.0            # Bend/turn cost weight
    
    # PathFinder parameters
    max_iterations: int = 50       # Max congestion iterations
    pres_fac_init: float = 0.5     # Initial present factor
    pres_fac_mult: float = 1.4     # Present factor multiplier (tightened for forced choke)
    hist_cost_step: float = 1.0    # History cost increment
    alpha: float = 2.0             # Congestion penalty exponent

@dataclass
class RouteRequest:
    """Single net routing request"""
    net_id: str
    source_pad: str    # Pad entry node ID
    sink_pad: str      # Pad exit node ID

@dataclass 
class RouteSegment:
    """Single route segment for visualization"""
    start_x: float
    start_y: float  
    end_x: float
    end_y: float
    layer: int
    width: float = 0.089  # 3.5mil in mm
    net_id: str = ""

@dataclass
class RouteResult:
    """Routing result for a single net"""
    net_id: str
    success: bool
    path: List[str] = field(default_factory=list)  # Node IDs in path
    edges: List[str] = field(default_factory=list) # Edge IDs used
    segments: List[RouteSegment] = field(default_factory=list)  # Coordinate segments for visualization
    cost: float = 0.0
    length_mm: float = 0.0
    via_count: int = 0

class RoutingResourceGraph:
    """FPGA-style Routing Resource Graph"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.nodes: Dict[str, RRGNode] = {}
        self.edges: Dict[str, RRGEdge] = {}
        # Memory-efficient adjacency using sets instead of lists (no duplicates)
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> edge_ids
        
        # Layer configuration
        self.layer_count = 6  # Will be updated from board data
        self.layer_directions = {}  # layer_index -> 'H'/'V'
        self._setup_layer_directions()
        
        # Statistics
        self.total_rails = 0
        self.total_buses = 0
        self.total_switches = 0
        
    def _setup_layer_directions(self):
        """Setup alternating H/V layer directions"""
        for i in range(self.layer_count):
            # Odd layers (In1.Cu=0, In3.Cu=2, etc.) = Horizontal
            # Even layers (In2.Cu=1, In4.Cu=3, etc.) = Vertical  
            self.layer_directions[i] = 'H' if i % 2 == 0 else 'V'
    
    def add_node(self, node: RRGNode) -> None:
        """Add node to RRG"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = set()  # Initialize as set
    
    def add_edge(self, edge: RRGEdge) -> None:
        """Add edge to RRG"""
        # Memory safety check - INCREASED limits for 16GB GPU ultra-dense routing
        if len(self.edges) > 50000000:  # 50M edge limit for 16GB ultra-dense grids
            raise MemoryError(f"RRG edge limit exceeded: {len(self.edges)} edges. "
                            "Consider reducing board size or grid resolution.")
        
        self.edges[edge.id] = edge
        
        # Add to adjacency sets - CRITICAL FIX for bidirectional connectivity
        # Using sets prevents duplicates and saves memory
        self.adjacency[edge.from_node].add(edge.id)
        
        # FPGA-style routing requires bidirectional traversal 
        # Add reverse adjacency so pathfinding can traverse both directions
        self.adjacency[edge.to_node].add(edge.id)
        
    def get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """Get neighboring (node_id, edge_id) pairs"""
        neighbors = []
        
        # Outgoing edges
        for edge_id in self.adjacency[node_id]:
            edge = self.edges[edge_id]
            neighbors.append((edge.to_node, edge_id))
            
        # Incoming edges (for bidirectional search)
        for edge_id, edge in self.edges.items():
            if edge.to_node == node_id:
                neighbors.append((edge.from_node, edge_id))
                
        return neighbors
    
    def manhattan_distance(self, node1_id: str, node2_id: str) -> float:
        """Calculate Manhattan distance between nodes"""
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return abs(n1.x - n2.x) + abs(n1.y - n2.y)
    
    def estimate_min_vias(self, node1_id: str, node2_id: str) -> int:
        """Estimate minimum vias needed between nodes"""
        n1 = self.nodes[node1_id]
        n2 = self.nodes[node2_id]
        return abs(n1.layer - n2.layer)
    
    def clear_usage(self):
        """Clear all usage counters"""
        for node in self.nodes.values():
            node.usage = 0
        for edge in self.edges.values():
            edge.usage = 0
    
    def get_overused_edges(self) -> List[str]:
        """Get list of overused edge IDs"""
        return [edge_id for edge_id, edge in self.edges.items() 
                if edge.usage > edge.capacity]
    
    def update_history_costs(self, overused_edges: List[str], step: float):
        """Update history costs for overused edges"""
        for edge_id in overused_edges:
            edge = self.edges[edge_id]
            excess = edge.usage - edge.capacity
            edge.history_cost += step * excess

class PathFinderRouter:
    """PathFinder negotiated congestion router with GPU wavefront pathfinding"""
    
    def __init__(self, rrg: RoutingResourceGraph):
        self.rrg = rrg
        self.config = rrg.config
        
        # Routing state
        self.net_routes: Dict[str, RouteResult] = {}
        self.current_iteration = 0
        self.present_factor = self.config.pres_fac_init
        
        # FIXED: Use parallel PathFinder instead of broken wavefront PathFinder
        from .gpu_pathfinder import GPUPathFinderRouter
        from .gpu_rrg import GPURoutingResourceGraph
        
        # Convert to GPU RRG and initialize parallel PathFinder
        self.gpu_rrg = GPURoutingResourceGraph(rrg, use_gpu=True)
        
        # CRITICAL: Register escape route nodes as tap nodes for PathFinder expansion
        self._register_escape_routes_as_tap_nodes()
        
        self.parallel_router = GPUPathFinderRouter(self.gpu_rrg, self.config)
        logger.info("PathFinder initialized with PARALLEL GPU PathFinder (FIXED VERSION)")
        
    def route_single_net(self, request: RouteRequest) -> RouteResult:
        """Route single net using GPU wavefront pathfinding with F.Cu escape routing"""
        
        # Check if this routing requires F.Cu escape routing
        source_node = self.rrg.nodes.get(request.source_pad)
        sink_node = self.rrg.nodes.get(request.sink_pad)
        
        if source_node and sink_node:
            source_is_fcu = source_node.layer == -2  # F.Cu layer
            sink_is_fcu = sink_node.layer == -2      # F.Cu layer
            
            if source_is_fcu or sink_is_fcu:
                logger.debug(f"F.Cu escape routing needed for net {request.net_id}: source_fcu={source_is_fcu}, sink_fcu={sink_is_fcu}")
                return self._route_with_escape_routing(request, source_is_fcu, sink_is_fcu)
        
        # Use direct GPU wavefront pathfinder for internal routing
        result = self.wavefront_router.route_single_net(request)
        
        return result
    
    def route_net(self, request: RouteRequest) -> RouteResult:
        """Route net - alias for route_single_net for compatibility"""
        return self.route_single_net(request)
    
    def _route_with_escape_routing(self, request: RouteRequest, source_is_fcu: bool, sink_is_fcu: bool) -> RouteResult:
        """Route net with F.Cu escape routing"""
        logger.info(f"Routing net {request.net_id} with F.Cu escape routing")
        
        # CRITICAL: Ensure wavefront grid is built before creating escape vias
        if not hasattr(self.wavefront_router, 'grid') or self.wavefront_router.grid is None:
            logger.info("Pre-building wavefront grid for escape routing...")
            self.wavefront_router.build_grid()
        
        # Get source and sink nodes
        source_node = self.rrg.nodes[request.source_pad]
        sink_node = self.rrg.nodes[request.sink_pad]
        
        # Find internal grid entry/exit points for F.Cu pads
        actual_source = request.source_pad
        actual_sink = request.sink_pad
        escape_segments = []
        
        if source_is_fcu:
            # Find nearest internal grid point for source escape
            escape_point = self._find_escape_point(source_node)
            if not escape_point:
                logger.error(f"Cannot find escape point for F.Cu source {request.source_pad}")
                return RouteResult(net_id=request.net_id, success=False)
            
            actual_source = escape_point
            escape_segments.append(('escape', request.source_pad, escape_point))
            logger.debug(f"Source escape: {request.source_pad} -> {escape_point}")
        
        if sink_is_fcu:
            # Find nearest internal grid point for sink entry
            entry_point = self._find_escape_point(sink_node)
            if not entry_point:
                logger.error(f"Cannot find entry point for F.Cu sink {request.sink_pad}")
                return RouteResult(net_id=request.net_id, success=False)
            
            actual_sink = entry_point
            escape_segments.append(('entry', entry_point, request.sink_pad))
            logger.debug(f"Sink entry: {entry_point} -> {request.sink_pad}")
        
        # Create modified request for internal grid routing
        internal_request = RouteRequest(
            net_id=request.net_id,
            source_pad=actual_source,
            sink_pad=actual_sink
        )
        
        # Add any pending escape vias to wavefront grid before routing
        self._add_pending_escape_vias_to_grid()
        
        # Route between internal grid points
        result = self.wavefront_router.route_single_net(internal_request)
        
        if result.success:
            # Add escape routing segments to the path
            full_path = []
            
            # Add source escape segment
            if source_is_fcu:
                full_path.append(request.source_pad)  # F.Cu pad
                full_path.append(actual_source)       # Internal grid point
            
            # Add internal routing path (skip first node if escape was added)
            if result.path:
                path_start = 1 if source_is_fcu else 0
                full_path.extend(result.path[path_start:])
            
            # Add sink entry segment  
            if sink_is_fcu and request.sink_pad not in full_path:
                full_path.append(request.sink_pad)    # F.Cu pad
            
            logger.info(f"F.Cu escape routing SUCCESS: net {request.net_id}, path length {len(full_path)}")
            
            # Find actual edge IDs for the full path (no more fake IDs)
            full_path_edges = []
            for i in range(len(full_path) - 1):
                from_node = full_path[i]
                to_node = full_path[i + 1]
                # Find actual edge ID instead of creating fake one
                actual_edge_id = self._find_edge_between_nodes(from_node, to_node)
                if actual_edge_id:
                    full_path_edges.append(actual_edge_id)
                else:
                    logger.debug(f"No edge found between {from_node} and {to_node} - skipping")
            
            return RouteResult(
                net_id=request.net_id,
                success=True,
                path=full_path,
                edges=full_path_edges,
                cost=result.cost + len(escape_segments) * 2,  # Add escape cost
                length_mm=result.length_mm,
                via_count=result.via_count + len(escape_segments)
            )
        else:
            logger.warning(f"F.Cu escape routing FAILED: internal routing failed for net {request.net_id}")
            # Even if internal routing failed, return escape routing segments for visualization
            # Store current escape vias before they get cleared
            current_escape_vias = getattr(self, '_pending_escape_vias', [])
            partial_result = self._create_partial_escape_route_result(request, source_is_fcu, sink_is_fcu, escape_segments, current_escape_vias)
            return partial_result
    
    def _find_escape_point(self, fcu_node) -> Optional[str]:
        """Create FPGA-style escape routing to nearest grid intersection"""
        # Find nearest grid intersection point for structured routing
        nearest_intersection = self._find_nearest_grid_intersection(fcu_node.x, fcu_node.y)
        
        if not nearest_intersection:
            logger.error(f"No grid intersection found for F.Cu pad at ({fcu_node.x:.2f}, {fcu_node.y:.2f})")
            return None
            
        via_x, via_y, layer = nearest_intersection
        
        # For FPGA-style routing, connect directly to grid intersection
        # Keep short F.Cu stub (vertical breakout preferred)
        directions = [
            (0, 1),   # Up (+Y) - preferred for top-side routing
            (0, -1),  # Down (-Y) - backup direction
            (1, 0),   # Right (+X) - secondary option  
            (-1, 0),  # Left (-X) - secondary option
        ]
        
        # Find best breakout direction toward grid intersection
        best_direction = None
        min_distance = float('inf')
        
        for dx, dy in directions:
            # Try 1mm breakout in this direction
            stub_distance = 1.0  # Shorter stub for denser routing
            test_x = fcu_node.x + dx * stub_distance  
            test_y = fcu_node.y + dy * stub_distance
            
            # Calculate distance to grid intersection
            distance = math.sqrt((test_x - via_x)**2 + (test_y - via_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_direction = (dx, dy)
        
        if best_direction:
            dx, dy = best_direction
            stub_x = fcu_node.x + dx * 1.0  # 1mm F.Cu stub
            stub_y = fcu_node.y + dy * 1.0
            
            # Create structured escape routing to grid intersection
            if self._is_escape_position_clear(stub_x, stub_y):
                # Create escape via at grid intersection
                via_id = f"fabric_entry_{fcu_node.id}_{via_x:.1f}_{via_y:.1f}"
                
                # Use existing grid intersection or create switch point
                intersection_node = self._get_or_create_grid_intersection(via_x, via_y, layer)
                
                if intersection_node:
                    # Create F.Cu breakout stub from pad to via 
                    stub_distance = math.sqrt((stub_x - fcu_node.x)**2 + (stub_y - fcu_node.y)**2)
                    fcu_trace_id = f"fcu_stub_{fcu_node.id}_to_grid"
                    fcu_trace = RRGEdge(
                        id=fcu_trace_id,
                        edge_type=EdgeType.ENTRY,  # F.Cu breakout
                        from_node=fcu_node.id,
                        to_node=intersection_node.id,
                        length_mm=stub_distance + min_distance,  # Total routing distance
                        base_cost=(stub_distance + min_distance) * 1.5  # Higher cost for F.Cu + via
                    )
                    self.rrg.add_edge(fcu_trace)
                    
                    logger.info(f"Connected F.Cu pad {fcu_node.id} to grid at ({via_x:.1f},{via_y:.1f}) via {stub_distance:.1f}mm stub")
                    return intersection_node.id
                
        logger.warning(f"Could not find clear escape position for F.Cu node {fcu_node.id}")
        return None
    
    def _find_nearest_grid_intersection(self, x: float, y: float) -> Optional[Tuple[float, float, int]]:
        """Find nearest grid intersection point for FPGA-style routing"""
        # Use 0.2mm grid pitch from config for dense backplane
        grid_pitch = 0.2  # Match dense pad spacing
        
        # Find nearest grid intersection coordinates
        grid_x = round(x / grid_pitch) * grid_pitch
        grid_y = round(y / grid_pitch) * grid_pitch
        
        # Use first internal routing layer (layer 0 = In1.Cu)
        target_layer = 0
        
        logger.debug(f"Mapped F.Cu pad at ({x:.2f},{y:.2f}) to grid intersection ({grid_x:.2f},{grid_y:.2f}) on layer {target_layer}")
        return (grid_x, grid_y, target_layer)
    
    def _get_or_create_grid_intersection(self, x: float, y: float, layer: int) -> Optional[RRGNode]:
        """Get existing grid intersection or create switch point with multi-layer via support"""
        # Support blind/buried vias at same coordinates - include layer in ID
        # This allows multiple via types at same (x,y): F.Cu->In1.Cu, In5.Cu->B.Cu, etc.
        switch_id = f"grid_switch_L{layer}_{x:.1f}_{y:.1f}"
        
        if switch_id in self.rrg.nodes:
            return self.rrg.nodes[switch_id]
        
        # Look for nearby routing nodes (rails/buses) to connect to
        # CRITICAL FIX: Search in adjacent grid cells to avoid self-connection
        nearby_nodes = []
        search_radius = self.config.grid_pitch * 1.5  # Search in adjacent cells
        
        for node_id, node in self.rrg.nodes.items():
            if node.layer == layer and node.node_type in [NodeType.RAIL, NodeType.BUS]:
                distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
                # CRITICAL: Avoid connecting to nodes at exactly same position (self-loops)
                if self.config.grid_pitch * 0.1 < distance <= search_radius:
                    nearby_nodes.append((node_id, node, distance))
        
        if not nearby_nodes:
            logger.warning(f"No routing nodes found near grid intersection ({x:.1f},{y:.1f}) layer {layer}")
            return None
        
        # Find closest routing node (but not at same position)
        nearest_node_id, nearest_node, _ = min(nearby_nodes, key=lambda n: n[2])
        
        # Create switch point at grid intersection
        switch_node = RRGNode(
            id=switch_id,
            node_type=NodeType.SWITCH,
            x=x,
            y=y,
            layer=layer,
            capacity=1
        )
        self.rrg.add_node(switch_node)
        
        # Connect switch to nearest routing node
        conn_id = f"switch_conn_{switch_id}_to_{nearest_node_id}"
        switch_edge = RRGEdge(
            id=conn_id,
            edge_type=EdgeType.SWITCH,
            from_node=switch_id,
            to_node=nearest_node_id,
            length_mm=math.sqrt((switch_node.x - nearest_node.x)**2 + (switch_node.y - nearest_node.y)**2),
            base_cost=self.config.k_via  # Via cost for grid entry
        )
        self.rrg.add_edge(switch_edge)
        
        # Also create reverse edge for bidirectional connectivity
        reverse_conn_id = f"switch_conn_{nearest_node_id}_to_{switch_id}"
        reverse_switch_edge = RRGEdge(
            id=reverse_conn_id,
            edge_type=EdgeType.SWITCH,
            from_node=nearest_node_id,
            to_node=switch_id,
            length_mm=math.sqrt((switch_node.x - nearest_node.x)**2 + (switch_node.y - nearest_node.y)**2),
            base_cost=self.config.k_via  # Via cost for grid entry
        )
        self.rrg.add_edge(reverse_switch_edge)
        
        # CRITICAL: Add the new switch node to the wavefront grid dynamically
        if hasattr(self.wavefront_router, 'add_node_to_grid'):
            grid_pos = self.wavefront_router.add_node_to_grid(switch_id, switch_node)
            if grid_pos:
                logger.info(f"Added grid intersection switch {switch_id} to wavefront grid at {grid_pos}")
            else:
                logger.error(f"Failed to add switch {switch_id} to wavefront grid")
        else:
            logger.warning("Wavefront router does not support dynamic node addition")
        
        logger.info(f"Created grid intersection switch {switch_id} connected to {nearest_node_id}")
        return switch_node
    
    def _create_partial_escape_route_result(self, request: RouteRequest, source_is_fcu: bool, sink_is_fcu: bool, escape_segments: List, escape_vias: List = None) -> RouteResult:
        """Create route result showing escape routing even when internal routing fails"""
        partial_path = []
        
        # Add escape routing segments to show F.Cu traces and vias
        if escape_vias is None:
            escape_vias = getattr(self, '_pending_escape_vias', [])
            
        logger.debug(f"Looking for escape vias in {len(escape_vias)} pending vias for net {request.net_id}")
        
        if source_is_fcu and escape_vias:
            # Find source escape via by checking if the source pad name is in via_id
            source_pad_name = request.source_pad.replace('pad_entry_', '')
            for via_id, escape_via in escape_vias:
                if source_pad_name in via_id:
                    partial_path.extend([request.source_pad, via_id])  # F.Cu pad -> escape via
                    logger.info(f"Added source escape: {request.source_pad} -> {via_id}")
                    break
        
        if sink_is_fcu and escape_vias:
            # Find sink escape via by checking if the sink pad name is in via_id
            sink_pad_name = request.sink_pad.replace('pad_entry_', '')
            for via_id, escape_via in escape_vias:
                if sink_pad_name in via_id and via_id not in partial_path:
                    partial_path.extend([via_id, request.sink_pad])  # Escape via -> F.Cu pad
                    logger.info(f"Added sink escape: {via_id} -> {request.sink_pad}")
                    break
        
        logger.info(f"Created partial escape route for {request.net_id}: {len(partial_path)} segments")
        
        # Find actual edge IDs for the path segments (no fake IDs)
        path_edges = []
        for i in range(len(partial_path) - 1):
            from_node = partial_path[i]
            to_node = partial_path[i + 1]
            # Find actual edge ID instead of creating fake one
            actual_edge_id = self._find_edge_between_nodes(from_node, to_node)
            if actual_edge_id:
                path_edges.append(actual_edge_id)
            else:
                logger.debug(f"No edge found between {from_node} and {to_node} for partial path")
        
        return RouteResult(
            net_id=request.net_id,
            success=False,  # Mark as FAILED - only complete routes should succeed
            path=partial_path,
            edges=path_edges,
            cost=len(escape_segments) * 2.0,
            length_mm=len(escape_segments) * 0.5,  # Escape distance
            via_count=len(escape_segments)
        )
        
    def _add_pending_escape_vias_to_grid(self):
        """Add pending escape vias to the wavefront grid after construction"""
        if not hasattr(self, '_pending_escape_vias'):
            logger.debug("No pending escape vias to add")
            return
            
        logger.info(f"Adding {len(self._pending_escape_vias)} pending escape vias to wavefront grid")
        
        if hasattr(self.wavefront_router, 'add_node_to_grid'):
            for via_id, escape_via in self._pending_escape_vias:
                grid_pos = self.wavefront_router.add_node_to_grid(via_id, escape_via)
                if grid_pos:
                    logger.info(f"Added escape via {via_id} to wavefront grid at {grid_pos}")
                else:
                    logger.warning(f"Failed to add escape via {via_id} to wavefront grid")
        else:
            logger.warning("Wavefront router does not support add_node_to_grid")
        
        # Clear pending vias
        self._pending_escape_vias = []
    
    def _connect_via_to_routing_grid(self, via_id: str, via_x: float, via_y: float):
        """Connect escape via to nearby routing grid nodes"""
        connections_made = 0
        max_search_distance = 2.0  # Search within 2mm radius
        
        # Find nearby routing grid nodes
        for node_id, node in self.rrg.nodes.items():
            if node.layer >= 0 and node.node_type in [NodeType.BUS, NodeType.RAIL]:
                # Calculate distance to via
                distance = math.sqrt((node.x - via_x)**2 + (node.y - via_y)**2)
                
                if distance <= max_search_distance:
                    # Create connection from via to routing node
                    conn_id = f"via_conn_{via_id}_to_{node_id}"
                    via_conn = RRGEdge(
                        id=conn_id,
                        edge_type=EdgeType.SWITCH,  # Via connection acts as a switch
                        from_node=via_id,
                        to_node=node_id,
                        length_mm=distance,
                        base_cost=distance * 0.5 + 1.0  # Via connection cost
                    )
                    self.rrg.add_edge(via_conn)
                    connections_made += 1
                    
                    # Limit connections to avoid too many
                    if connections_made >= 4:
                        break
        
        logger.debug(f"Connected escape via {via_id} to {connections_made} routing grid nodes")
    
    def _is_escape_position_clear(self, x: float, y: float) -> bool:
        """Check if escape via position is clear of obstacles"""
        # For now, assume all positions are clear
        # TODO: Check for component keepouts, existing vias, etc.
        return True
    
    def _is_node_accessible(self, node_id: str) -> bool:
        """Check if node is accessible for escape routing"""
        # For now, consider all internal grid nodes accessible
        # Could add more sophisticated checking later (obstacles, existing routes)
        return True
    
    def _create_route_result(self, net_id: str, success: bool, path: List[str], cost: float) -> RouteResult:
        """Create route result from path"""
        if not success or len(path) < 2:
            return RouteResult(net_id=net_id, success=success)
        
        # Calculate edges used
        edges_used = []
        total_length = 0.0
        via_count = 0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Find edge between these nodes
            edge_id = self._find_edge_between_nodes(from_node, to_node)
            if edge_id:
                edges_used.append(edge_id)
                edge = self.rrg.edges[edge_id]
                total_length += edge.length_mm
                
                if edge.edge_type in [EdgeType.SWITCH, EdgeType.ENTRY, EdgeType.EXIT]:
                    via_count += 1
        
        return RouteResult(
            net_id=net_id,
            success=success,
            path=path,
            edges=edges_used,
            cost=cost,
            length_mm=total_length,
            via_count=via_count
        )
    
    def _find_edge_between_nodes(self, from_id: str, to_id: str) -> Optional[str]:
        """Find edge between two nodes"""
        for edge_id in self.rrg.adjacency[from_id]:
            edge = self.rrg.edges[edge_id]
            if edge.to_node == to_id:
                return edge_id
        
        # Try reverse direction
        for edge_id in self.rrg.adjacency[to_id]:
            edge = self.rrg.edges[edge_id]
            if edge.to_node == from_id:
                return edge_id
                
        return None
    
    def claim_route(self, result: RouteResult):
        """Claim resources for a successful route"""
        # Handle edges with safety check
        missing_edges = []
        for edge_id in result.edges:
            if edge_id in self.rrg.edges:
                edge = self.rrg.edges[edge_id]
                edge.usage += 1
            else:
                missing_edges.append(edge_id)
        
        if missing_edges:
            logger.warning(f"Route claiming: {len(missing_edges)} missing edges out of {len(result.edges)} total")
            logger.debug(f"Missing edge IDs: {missing_edges[:3]}...")
            
        # Handle nodes
        for node_id in result.path:
            if node_id in self.rrg.nodes:
                node = self.rrg.nodes[node_id]
                if node.node_type in [NodeType.RAIL, NodeType.BUS]:
                    node.usage += 1
            else:
                logger.debug(f"Missing node in claim_route: {node_id}")
    
    def route_all_nets(self, requests: List[RouteRequest]) -> Dict[str, RouteResult]:
        """Route all nets using PARALLEL PathFinder with proper tap node expansion"""
        logger.info(f"FIXED RRG PATHFINDER: Using parallel PathFinder for {len(requests)} nets")
        
        # Use parallel PathFinder instead of broken wavefront
        routes = self.parallel_router.route_all_nets_parallel(requests)
        
        # Convert parallel PathFinder results to RRG format
        results = {}
        for net_id, path_nodes in routes.items():
            if path_nodes:  # Successfully routed
                # Convert path to segments for display
                segments = self._convert_path_to_segments(path_nodes, net_id)
                via_count = len([s for s in segments if s.layer != segments[0].layer if segments])
                total_length = sum(s.length_mm for s in segments)
                
                results[net_id] = RouteResult(
                    success=True,
                    segments=segments,
                    total_length=total_length,
                    via_count=via_count,
                    layer_changes=via_count
                )
                logger.info(f"PARALLEL SUCCESS: Net {net_id} routed with {len(path_nodes)} nodes -> {len(segments)} segments")
            else:  # Failed to route
                results[net_id] = RouteResult(
                    success=False,
                    segments=[],
                    total_length=0.0,
                    via_count=0,
                    layer_changes=0
                )
        
        # Return results from parallel PathFinder
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(f"PathFinder complete: {success_count}/{len(requests)} nets routed")
        
        return results
    
    def _register_escape_routes_as_tap_nodes(self):
        """Register all escape route nodes as tap nodes for PathFinder expansion"""
        if not self.gpu_rrg:
            logger.error("Cannot register escape routes: GPU RRG not initialized")
            return

        logger.info("RRG: Registering escape route nodes as tap nodes for PathFinder...")
        
        # Find all escape route nodes (vertical and dogleg endpoints)
        tap_count = 0
        for node_id in self.gpu_rrg.node_id_to_idx.keys():
            if node_id.startswith('fcu_escape_v_') or node_id.startswith('fcu_escape_d_'):
                # Get node index
                node_idx = self.gpu_rrg.node_id_to_idx[node_id]
                
                # Register as tap node
                self.gpu_rrg.tap_nodes[node_id] = node_idx
                tap_count += 1

        logger.info(f"RRG TAP NODE REGISTRATION: Registered {tap_count} escape route nodes as tap nodes")
    
    def _convert_path_to_segments(self, path_nodes: List, net_id: str) -> List:
        """Convert PathFinder path nodes to display segments"""
        if not path_nodes or len(path_nodes) < 2:
            return []

        segments = []
        logger.debug(f"SEGMENT CONVERSION: Converting {len(path_nodes)} path nodes for {net_id}")
        
        # Create simple segments between consecutive nodes  
        for i in range(len(path_nodes) - 1):
            # Create basic segment (simplified for now)
            segment = {
                'start_x': 0.0,
                'start_y': 0.0, 
                'end_x': 0.4,
                'end_y': 0.0,
                'layer': 0,
                'length_mm': 0.4
            }
            segments.append(segment)
        
        return segments


def preflight_graph(graph_state) -> bool:
    """
    STEP 4: Preflight validation function for GraphState
    
    Validates graph integrity before pad mapping to catch issues early.
    This is part of the pipeline: lattice + CSR → preflight → pad mapping → routing
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[PREFLIGHT] Starting graph validation on graph_state id = {id(graph_state)}...")

    # DEBUG: Show what attributes the graph_state actually has
    logger.info(f"[PREFLIGHT] DEBUG: graph_state attributes: {dir(graph_state)}")
    logger.info(f"[PREFLIGHT] DEBUG: hasattr(graph_state, 'indptr'): {hasattr(graph_state, 'indptr')}")
    if hasattr(graph_state, 'indptr'):
        logger.info(f"[PREFLIGHT] DEBUG: graph_state.indptr: {graph_state.indptr}")
        logger.info(f"[PREFLIGHT] DEBUG: graph_state.indptr is None: {graph_state.indptr is None}")
        if graph_state.indptr is not None:
            logger.info(f"[PREFLIGHT] DEBUG: graph_state.indptr type: {type(graph_state.indptr)}")

    # Basic validation - check if we have required arrays
    if not hasattr(graph_state, 'indptr') or graph_state.indptr is None:
        logger.error("[PREFLIGHT] FAILED: Missing indptr array")
        return False
        
    if not hasattr(graph_state, 'indices') or graph_state.indices is None:
        logger.error("[PREFLIGHT] FAILED: Missing indices array")
        return False
        
    if not hasattr(graph_state, 'weights') or graph_state.weights is None:
        logger.error("[PREFLIGHT] FAILED: Missing weights array")
        return False
    
    # Check array sizes match
    if len(graph_state.indices) != len(graph_state.weights):
        logger.error(f"[PREFLIGHT] FAILED: indices({len(graph_state.indices)}) != weights({len(graph_state.weights)})")
        return False
    
    # Check indptr bounds
    if len(graph_state.indptr) == 0:
        logger.error("[PREFLIGHT] FAILED: Empty indptr array")
        return False
        
    # Log basic stats
    num_nodes = len(graph_state.indptr) - 1 if len(graph_state.indptr) > 0 else 0
    num_edges = len(graph_state.indices)
    
    logger.info(f"[PREFLIGHT] Graph has {num_nodes} nodes, {num_edges} edges")
    logger.info(f"[PREFLIGHT] indptr[0]={graph_state.indptr[0]}, indptr[-1]={graph_state.indptr[-1]}")
    
    # Validate indptr bounds
    if graph_state.indptr[-1] != num_edges:
        logger.error(f"[PREFLIGHT] FAILED: indptr[-1]={graph_state.indptr[-1]} != num_edges={num_edges}")
        return False
    
    logger.info("[PREFLIGHT] All basic validations passed")
    return True