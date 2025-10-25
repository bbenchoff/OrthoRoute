"""
Clean Manhattan Router Interface - GPU-First Rewrite

Maintains plugin compatibility while using clean GPU-accelerated implementation.
Delegates to LatticeBuilder + FastGPUPathFinder for actual routing.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime

from ...domain.models.board import Board, Net, Pad, Bounds, Coordinate
from ...domain.models.routing import Route, Segment, Via, RoutingResult, RoutingStatistics, SegmentType, ViaType
from ...domain.models.constraints import DRCConstraints
from ...domain.services.routing_engine import RoutingEngine, RoutingStrategy
from ...application.interfaces.gpu_provider import GPUProvider

from .unified_pathfinder import UnifiedPathFinder, PathFinderConfig
from .types import Pad as RRGPad
from dataclasses import dataclass
from typing import List, Any

logger = logging.getLogger(__name__)


@dataclass 
class RouteResult:
    """Simple route result for internal use"""
    net_id: str
    success: bool
    path: List[int]
    segments: List[Any]
    vias: List[Any] 
    routing_time: float
    path_length: int


class ManhattanRRGRoutingEngine(RoutingEngine):
    """Clean Manhattan routing engine using GPU-accelerated 3D lattice PathFinder"""
    
    def __init__(self, constraints: DRCConstraints, gpu_provider: Optional[GPUProvider] = None, pathfinder: Optional[UnifiedPathFinder] = None):
        """Initialize clean GPU-first Manhattan routing engine
        
        Args:
            constraints: DRC constraints
            gpu_provider: Optional GPU provider
            pathfinder: Pre-created UnifiedPathFinder instance (recommended to avoid "second instance" bug)
        """
        super().__init__(constraints)
        
        self.gpu_provider = gpu_provider
        self.board: Optional[Board] = None
        self.progress_callback = None
        
        # INSTANCE BUG FIX: Accept pre-created UnifiedPathFinder instance
        self.pathfinder: Optional[UnifiedPathFinder] = pathfinder
        
        # Results storage
        self.routing_results: Dict[str, RouteResult] = {}
        self.routed_nets: Set[str] = set()
        
        logger.info("Clean Manhattan RRG routing engine initialized")
    
    def _find_nearest_node(self, x: float, y: float, layer: int = 0) -> Optional[str]:
        """Find the nearest routing node using UNIFIED PathFinder's optimized lookup"""
        if not self.pathfinder:
            logger.error("PathFinder not available")
            return None
        
        # Check for exact pad match first
        pad_node_id = f"pad_*_{x:.1f}_{y:.1f}"
        for node_id in self.pathfinder.nodes.keys():
            if node_id.startswith('pad_') and f"_{x:.1f}_{y:.1f}" in node_id:
                logger.debug(f"Found exact pad match: {node_id}")
                return node_id
        
        # Use PathFinder's optimized spatial lookup
        nearest_rail = self.pathfinder._find_nearest_rail_fast(x, y, layer, 10.0)  # 10mm max
        
        if nearest_rail:
            logger.debug(f"Found nearest rail: {nearest_rail} for ({x}, {y}, {layer})")
            return nearest_rail
        else:
            logger.warning(f"No node found within 10mm of ({x}, {y}, {layer})")
            return None
    
    def initialize(self, board: Board) -> bool:
        """Initialize router with board data"""
        try:
            logger.info(f"Initializing clean router for board: {board.name}")
            
            self.board = board
            
            # Convert domain pads to algorithm pads
            algorithm_board = self._create_algorithm_board(board)
            
            # INSTANCE BUG FIX: Use passed-in UnifiedPathFinder instance or create if none provided
            if self.pathfinder is None:
                logger.warning("No UnifiedPathFinder instance provided - creating new one (NOT RECOMMENDED)")
                logger.info("Initializing Unified High-Performance PathFinder...")
                
                # Create optimized PathFinder config from DRC constraints
                config = PathFinderConfig(
                    initial_pres_fac=0.5,
                    pres_fac_mult=1.3,
                    acc_fac=1.0,
                    max_iterations=6,  # Reduced for speed with adaptive early stopping
                    grid_pitch=0.4,
                    max_search_nodes=30000,  # Optimized limit
                    mode="multi_roi_bidirectional",  # Use Multi-ROI Parallel Bidirectional A* PathFinder
                    roi_parallel=True,  # Enable K concurrent ROI processing

                    # Performance & Convergence optimizations
                    delta_multiplier=5.0,  # Start with 5x grid pitch for speed (4x-6x range)
                    adaptive_delta=True,   # Enable adaptive delta tuning
                    congestion_cost_mult=1.3,  # Enhanced congestion penalty (vs default 1.2)

                    # GPU Kernel & Memory optimizations
                    enable_memory_compaction=True,  # Compact ROI arrays for coalesced access
                    memory_alignment=128,           # 128-byte alignment for optimal coalescing
                    enable_profiling=False,         # Enable for Nsight profiling (set True for analysis)
                    warp_analysis=False,            # Enable for warp divergence analysis

                    # PHASE 3: Micro-batch negotiation parameters
                    use_micro_batch_negotiation=True,  # Enable micro-batch mode with gentler pressure
                    micro_batch_size=16,               # Process 16 nets per micro-batch
                    micro_batch_pres_fac_init=0.5,     # Start with gentle pressure (same as initial_pres_fac)
                    micro_batch_pres_fac_mult=1.5,     # Gentler escalation (vs 2.0 standard)

                    # Instrumentation & Logging (enabled by default)
                    enable_instrumentation=True,   # Enable detailed metrics collection and CSV export
                    csv_export_path="pathfinder_metrics.csv",
                    log_iteration_details=True,    # Log detailed iteration metrics
                    log_roi_statistics=True        # Log ROI batch statistics
                )
                
                self.pathfinder = UnifiedPathFinder(config=config, use_gpu=True)  # Enable GPU acceleration
            else:
                logger.info(f"Using pre-created UnifiedPathFinder instance: {getattr(self.pathfinder, '_instance_tag', 'NO_TAG')}")
            
            # Build routing lattice (consolidated function)
            if not self.pathfinder.build_routing_lattice(algorithm_board):
                logger.error("Failed to build routing lattice")
                return False
            
            logger.info("Clean router initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize clean router: {e}")
            return False
    
    def _create_algorithm_board(self, domain_board: Board):
        """Convert domain Board to algorithm-compatible board with proper Pad types"""
        from .types import Pad as AlgorithmPad
        from ...domain.models.board import Bounds
        
        # Create algorithm board mock
        class AlgorithmBoard:
            def __init__(self):
                self.name = domain_board.name
                self.layer_count = domain_board.layer_count
                self.pads = []
                # Copy airwires if available
                self._airwires = getattr(domain_board, '_airwires', [])
                # Try to get KiCad-calculated bounds if available
                self._kicad_bounds = getattr(domain_board, '_kicad_bounds', None)
            
            def get_bounds(self):
                """Return KiCad-calculated bounds if available, otherwise calculate from components"""
                if self._kicad_bounds:
                    min_x, min_y, max_x, max_y = self._kicad_bounds
                    return Bounds(min_x, min_y, max_x, max_y)
                # Fallback: calculate from pads (shouldn't be used if KiCad bounds available)
                if not self.pads:
                    return Bounds(0, 0, 100, 100)
                min_x = min(pad.x_mm for pad in self.pads)
                max_x = max(pad.x_mm for pad in self.pads)
                min_y = min(pad.y_mm for pad in self.pads)
                max_y = max(pad.y_mm for pad in self.pads)
                return Bounds(min_x, min_y, max_x, max_y)
        
        algorithm_board = AlgorithmBoard()
        
        # Convert domain nets to algorithm pads
        for net in domain_board.nets:
            for pad in net.pads:
                # Convert domain pad to algorithm pad
                algorithm_pad = AlgorithmPad(
                    net_name=net.name,
                    x_mm=pad.position.x,
                    y_mm=pad.position.y,
                    width_mm=pad.size[0],
                    height_mm=pad.size[1],
                    layer_set='THRU',  # Assume through-hole for now
                    is_through_hole=True
                )
                algorithm_board.pads.append(algorithm_pad)
        
        logger.info(f"Converted {len(algorithm_board.pads)} domain pads to algorithm pads")
        logger.info(f"Copied {len(algorithm_board._airwires)} airwires to algorithm board")
        if algorithm_board._kicad_bounds:
            logger.info(f"Using KiCad geometric bounds: {algorithm_board._kicad_bounds}")
        return algorithm_board
    
    def route_all_nets(self, nets: List[Net], 
                      strategy: Optional[RoutingStrategy] = None,
                      timeout_per_net: float = 30.0,
                      max_iterations: int = 10,
                      total_timeout: Optional[float] = None,
                      **kwargs) -> RoutingResult:
        """Route all nets using clean PathFinder implementation"""
        
        if not self.pathfinder:
            logger.error("PathFinder not initialized - call initialize() first")
            return RoutingResult(
                success=False,
                error_message="PathFinder not initialized",
                execution_time=0.0,
                algorithm_used=self.strategy
            )
        
        logger.info(f"Routing {len(nets)} nets with clean PathFinder")
        start_time = time.time()
        
        # Convert nets to route requests
        route_requests = []
        for net in nets:
            if len(net.pads) >= 2:
                # Simple approach: route between first two pads
                source_pad = net.pads[0]
                sink_pad = net.pads[1]
                
                # Handle both domain and algorithm pad types
                if hasattr(source_pad, 'x_mm'):
                    # Algorithm pad type
                    source_x, source_y = source_pad.x_mm, source_pad.y_mm
                    sink_x, sink_y = sink_pad.x_mm, sink_pad.y_mm
                else:
                    # Domain pad type - extract from position
                    source_x, source_y = source_pad.position.x, source_pad.position.y
                    sink_x, sink_y = sink_pad.position.x, sink_pad.position.y
                
                # Find nearest nodes in the routing graph
                source_node_id = self._find_nearest_node(source_x, source_y, 0)
                sink_node_id = self._find_nearest_node(sink_x, sink_y, 0)
                
                # Only add request if both nodes found
                if source_node_id and sink_node_id:
                    route_requests.append((net.name, source_node_id, sink_node_id))
                else:
                    logger.warning(f"Net {net.name}: Could not find routing nodes for pads at ({source_x}, {source_y}) and ({sink_x}, {sink_y})")
        
        logger.info(f"Created {len(route_requests)} route requests")
        
        # Route using PathFinder
        try:
            import traceback
            logger.info(f"About to call route_multiple_nets with {len(route_requests)} requests")
            paths = self.pathfinder.route_multiple_nets(route_requests)
            
            # Convert results
            success_count = 0
            for net_name, path in paths.items():
                if len(path) > 0:
                    success_count += 1
                    self.routed_nets.add(net_name)
                    
                    # Store route result
                    self.routing_results[net_name] = RouteResult(
                        net_id=net_name,
                        success=True,
                        path=path,
                        segments=[],  # Will populate if needed for visualization
                        vias=[],
                        routing_time=0.0,
                        path_length=len(path)
                    )
            
            elapsed_time = time.time() - start_time
            success_rate = success_count / len(nets) if nets else 0.0
            
            logger.info(f"Clean PathFinder routing complete: {success_count}/{len(nets)} nets routed in {elapsed_time:.2f}s (success rate: {success_rate:.1%})")
            
            return RoutingResult(
                success=success_count > 0,
                error_message=None if success_count > 0 else "No nets could be routed",
                execution_time=elapsed_time,
                algorithm_used=self.strategy
            )
            
        except Exception as e:
            logger.error(f"Clean PathFinder routing failed: {e}")
            logger.error("FULL STACK TRACE:")
            traceback.print_exc()
            logger.error("END STACK TRACE")
            
            # Log detailed information about the error
            if "unhashable type" in str(e):
                logger.error("NDARRAY ERROR DETECTED - This is the exact location of the ndarray hashability error!")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error args: {e.args}")
            
            return RoutingResult(
                success=False,
                error_message=str(e),
                execution_time=0.0,
                algorithm_used=self.strategy
            )
    
    def get_routed_tracks(self) -> List[Dict[str, Any]]:
        """Get routed tracks using UNIFIED PathFinder visualization"""
        if not self.pathfinder or not self.routing_results:
            logger.info("No routing results to visualize")
            return []
        
        try:
            # Convert paths to visualization data using unified implementation
            paths = {net_id: result.path for net_id, result in self.routing_results.items()}
            tracks = self.pathfinder.get_route_visualization_data(paths)
            
            logger.info(f"Generated {len(tracks)} track segments for visualization")
            return tracks
            
        except Exception as e:
            logger.error(f"Failed to generate visualization tracks: {e}")
            return []
    
    def get_routing_statistics(self) -> RoutingStatistics:
        """Get routing statistics"""
        nets_attempted = len(self.routing_results) if self.routing_results else 0
        nets_routed = len(self.routed_nets)
        
        return RoutingStatistics(
            nets_attempted=nets_attempted,
            nets_routed=nets_routed,
            nets_failed=nets_attempted - nets_routed,
            total_vias=sum(len(result.vias) for result in self.routing_results.values()),
            total_time=sum(result.routing_time for result in self.routing_results.values()),
            algorithm_used=self.strategy
        )
    
    def cleanup(self):
        """Clean up resources"""
        self.routing_results.clear()
        self.routed_nets.clear()
        self.pathfinder = None
        logger.info("Clean router cleanup complete")
    
    # Abstract method implementations for RoutingEngine
    def clear_routes(self):
        """Clear all routing results"""
        self.routing_results.clear()
        self.routed_nets.clear()
    
    def get_routed_vias(self) -> List[Any]:
        """Get routed vias"""
        vias = []
        for result in self.routing_results.values():
            vias.extend(result.vias)
        return vias
    
    def route_net(self, net: Net) -> bool:
        """Route a single net"""
        if not self.pathfinder or len(net.pads) < 2:
            return False
        
        source_pad = net.pads[0]
        sink_pad = net.pads[1]
        
        # Find nearest nodes in the routing graph
        source_node_id = self._find_nearest_node(source_pad.x_mm, source_pad.y_mm, 0)
        sink_node_id = self._find_nearest_node(sink_pad.x_mm, sink_pad.y_mm, 0)
        
        if not source_node_id or not sink_node_id:
            logger.warning(f"Net {net.name}: Could not find routing nodes for single net routing")
            return False
        
        path = self.pathfinder.route_net(source_node_id, sink_node_id)
        if path and len(path) > 0:
            self.routing_results[net.name] = RouteResult(
                net_id=net.name,
                success=True,
                path=path,
                segments=[],
                vias=[],
                routing_time=0.0,
                path_length=len(path)
            )
            self.routed_nets.add(net.name)
            return True
        return False
    
    def route_two_pads(self, pad1: Any, pad2: Any) -> bool:
        """Route between two pads"""
        if not self.pathfinder:
            return False
        
        # Use the same naming convention as lattice builder - rail_v format
        net_name = getattr(pad1, 'net_name', 'TEMP')
        
        # Find nearest nodes in the routing graph  
        source_node_id = self._find_nearest_node(pad1.x_mm, pad1.y_mm, 0)
        sink_node_id = self._find_nearest_node(pad2.x_mm, pad2.y_mm, 0)
        
        if not source_node_id or not sink_node_id:
            logger.warning(f"Two-pad routing: Could not find routing nodes for pads at ({pad1.x_mm}, {pad1.y_mm}) and ({pad2.x_mm}, {pad2.y_mm})")
            return False
        
        path = self.pathfinder.route_net(source_node_id, sink_node_id)
        
        if path and len(path) > 0:
            # Store result for visualization
            self.routing_results[f"TEMP_{net_name}"] = RouteResult(
                net_id=f"TEMP_{net_name}",
                success=True,
                path=path,
                segments=[],
                vias=[],
                routing_time=0.0,
                path_length=len(path)
            )
            self.routed_nets.add(f"TEMP_{net_name}")
            return True
        
        return False
    
    @property
    def strategy(self) -> str:
        """Return routing strategy name"""
        return "MANHATTAN_PATHFINDER"
    
    def supports_gpu(self) -> bool:
        """Check if GPU acceleration is supported"""
        return True


# Compatibility alias - plugins might expect this name
class ManhattanRouterRRG(ManhattanRRGRoutingEngine):
    """Compatibility alias for plugin integration"""
    
    def __init__(self, config=None):
        # Convert config to constraints if needed
        constraints = DRCConstraints() if not config else config
        super().__init__(constraints)
        
        # Additional compatibility attributes
        self.routing_config = config
        self.verification_results = {'overall_pass': True}
        
        logger.info("Manhattan router RRG compatibility wrapper initialized")