"""
Data types and structures for Manhattan routing.
Also provides light runtime-checked interfaces (BoardLike, Bounds)
so this package doesn't depend on a monorepo.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Protocol, runtime_checkable
from dataclasses import dataclass

# Occupancy kind constants
FREE = 0
PAD = 1
TRACE = 2
KEEPOUT = 3

# Unit conversion constant
IU_PER_MM = 1_000_000.0

# --- Light interfaces for plugin-facing code ---------------------------------
@dataclass
class Bounds:
    """Axis-aligned board bounds in millimeters."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

@runtime_checkable
class BoardLike(Protocol):
    """Minimal board protocol the router expects at runtime.
    Any object with these attributes/methods will work (e.g., KiCad adapter).
    """
    layer_count: int
    def get_bounds(self) -> Tuple[float, float, float, float]: ...
    def get_all_pads(self) -> List['Pad']: ...

@dataclass
class Pad:
    """Normalized pad representation"""
    net_name: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    layer_set: Union[Set[str], str]  # Set of layer names or "THRU"
    is_through_hole: bool

@dataclass
class Via:
    """Via structure"""
    x: float
    y: float
    from_layer: int
    to_layer: int
    size: float
    drill: float
    net_id: int

@dataclass
class Track:
    """Track/trace structure"""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: int
    width: float
    net_id: int

@dataclass
class RouteResult:
    """Routing result"""
    success: bool
    tracks: List[Track]
    vias: List[Via]
    routed_nets: int
    failed_nets: int
    stats: Dict[str, Any]

@dataclass
class RoutingConfig:
    """Routing configuration for Manhattan routing"""
    grid_pitch: float = 0.40  # mm - per spec
    track_width: float = 0.0889  # mm - 3.5mil per spec
    clearance: float = 0.0889  # mm - 3.5mil spacing per spec
    via_drill: float = 0.15  # mm - per spec
    via_diameter: float = 0.25  # mm - per spec
    bend_penalty: int = 2
    via_cost: int = 10
    expansion_margin: float = 3.0  # mm
    max_ripups_per_net: int = 2
    max_global_failures: int = 100
    timeout_per_net: float = 0.1
    
    # Manhattan routing specific
    num_internal_layers: int = 11  # In1-In10 + B.Cu per spec
    use_blind_buried_vias: bool = True
    
    @property
    def via_size(self) -> float:
        """Alias for via_diameter for compatibility"""
        return self.via_diameter

@dataclass 
class FabricSegment:
    """Fabric network segment"""
    segment_id: str
    layer_name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    claimed_by: Optional[str] = None  # Net name that claimed this segment
    
    def length(self) -> float:
        """Calculate segment length"""
        dx = self.end_x - self.start_x
        dy = self.end_y - self.start_y
        return (dx*dx + dy*dy) ** 0.5

@dataclass
class FabricNode:
    """Fabric network node"""
    node_id: str
    x: float
    y: float
    layer_name: str

@dataclass
class FabricNetwork:
    """Fabric routing network for internal layers"""
    segments: Dict[str, FabricSegment]
    nodes: Dict[str, FabricNode]
    
    def get_available_segments(self, layer_name: str) -> List[FabricSegment]:
        """Get unclaimed segments on specified layer"""
        return [seg for seg in self.segments.values() 
                if seg.layer_name == layer_name and seg.claimed_by is None]
    
    def claim_segment(self, segment_id: str, net_name: str) -> bool:
        """Claim a segment for a net"""
        if segment_id in self.segments and self.segments[segment_id].claimed_by is None:
            self.segments[segment_id].claimed_by = net_name
            return True
        return False
    
    def release_segments(self, net_name: str) -> int:
        """Release all segments claimed by a net"""
        released = 0
        for seg in self.segments.values():
            if seg.claimed_by == net_name:
                seg.claimed_by = None
                released += 1
        return released