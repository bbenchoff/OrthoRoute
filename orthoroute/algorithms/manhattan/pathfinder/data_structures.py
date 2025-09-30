"""
PathFinder Data Structures

Core data structures used throughout the PathFinder routing system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class Portal:
    """Represents a connection point between pad and routing grid."""
    x: float
    y: float
    layer: int
    net: str
    pad_layer: int


@dataclass
class EdgeRec:
    """Legacy edge record structure - kept for compatibility."""
    __slots__ = ("usage", "owners", "pres_cost", "edge_history",
                 "owner_net", "taboo_until_iter", "historical_cost")

    def __init__(self) -> None:
        """Initialize edge record with default values for PathFinder negotiation."""
        self.usage = 0
        self.owners: Set[str] = set()
        self.pres_cost = 0.0
        self.edge_history = 0.0
        self.owner_net: Optional[str] = None
        self.taboo_until_iter = -1
        self.historical_cost = 0.0


@dataclass
class Geometry:
    """Geometry container for routing results."""
    tracks: list = field(default_factory=list)
    vias: list = field(default_factory=list)


def canonical_edge_key(layer_id: int, u1: int, v1: int, u2: int, v2: int) -> Tuple[int, int, int, int, int]:
    """Create canonical edge key for consistent storage and lookup.

    Args:
        layer_id: Layer index
        u1, v1: First node coordinates
        u2, v2: Second node coordinates

    Returns:
        Tuple of (layer_id, min_u, min_v, max_u, max_v) for consistent ordering
    """
    if (u1, v1) > (u2, v2):
        u1, v1, u2, v2 = u2, v2, u1, v1
    return (layer_id, u1, v1, u2, v2)