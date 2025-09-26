"""
Geometry Registry - Single Source of Truth for Lattice Operations

Consolidates all edge keys, lattice coordinates, and geometry operations into one module.
Eliminates mixed key formats and provides dense arrays for GPU acceleration.
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional, List, Set
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Canonical edge key: (layer_id, U1, V1, U2, V2) with (U2,V2)<(U1,V1)
Key5 = Tuple[int, int, int, int, int]

@dataclass(frozen=True)
class LayerInfo:
    """Layer metadata with routing orientation"""
    name: str
    layer_id: int
    orientation: str  # "H" (horizontal), "V" (vertical), or "VIA"
    kicad_name: str

@dataclass
class GridInfo:
    """Grid configuration parameters"""
    pitch_mm: float
    bbox_mm: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    u0_mm: float  # Grid origin X
    v0_mm: float  # Grid origin Y
    u_steps: int  # Number of U steps (x_steps)
    v_steps: int  # Number of V steps (y_steps)

class LatticeRegistry:
    """
    Single source of truth for all lattice geometry operations.

    Eliminates:
    - Mixed edge key formats (float vs int coordinates)
    - Dict-based edge tracking in hot loops
    - Inconsistent coordinate transformations
    - GPU/CPU data synchronization issues
    """

    _instance: Optional["LatticeRegistry"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self._frozen = False
        self.layers: Dict[int, LayerInfo] = {}
        self.grid: Optional[GridInfo] = None

        # Edge storage during construction
        self._key5_to_eid: Dict[Key5, int] = {}
        self._eid_to_key5: List[Key5] = []
        self._via_edge_ids: Set[int] = set()           # track via edges during construction

        # Dense arrays (allocated after freeze)
        self.E: int = 0
        self.eid_to_key5: Optional[np.ndarray] = None    # (E,5) int32
        self.base_cost: Optional[np.ndarray] = None      # (E,) float32
        self.owner: Optional[np.ndarray] = None          # (E,) int32  (-1=free)
        self.hist_cost: Optional[np.ndarray] = None      # (E,) float32
        self.present_cost: Optional[np.ndarray] = None   # (E,) float32
        self.blocked: Optional[np.ndarray] = None        # (E,) uint8
        self.e_is_via: Optional[np.ndarray] = None       # (E,) uint8

        # Per-net tracking
        self.net_to_idx: Dict[str, int] = {}             # net_name -> dense index
        self.idx_to_net: List[str] = []                  # dense index -> net_name
        self.fixed_eids_by_net: Dict[int, Set[int]] = {} # net_idx -> set of fixed edge IDs

    def reset(self):
        """Reset singleton for testing"""
        LatticeRegistry._instance = None

    # ---- Configuration Phase ----

    def set_grid(self, pitch_mm: float, bbox_mm: Tuple[float, float, float, float],
                 origin_mm: Tuple[float, float], u_steps: int, v_steps: int):
        """Configure grid parameters with lattice dimensions"""
        assert not self._frozen, "Cannot modify frozen registry"
        self.grid = GridInfo(pitch_mm, bbox_mm, origin_mm[0], origin_mm[1], u_steps, v_steps)
        logger.info(f"[REGISTRY] Grid configured: pitch={pitch_mm}mm, bbox={bbox_mm}, {u_steps}×{v_steps} lattice")

    def add_layer(self, layer_id: int, name: str, kicad_name: str, orientation: str):
        """Add layer with routing orientation"""
        assert not self._frozen, "Cannot modify frozen registry"
        assert orientation in ("H", "V", "VIA"), f"Invalid orientation: {orientation}"
        self.layers[layer_id] = LayerInfo(name, layer_id, orientation, kicad_name)
        logger.info(f"[REGISTRY] Added layer {layer_id}: {name} ({orientation})")

    # ---- Coordinate Transformations ----

    def mm_to_uv(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert world coordinates to lattice grid coordinates"""
        assert self.grid is not None, "Grid not configured"
        pitch = self.grid.pitch_mm
        U = int(round((x_mm - self.grid.u0_mm) / pitch))
        V = int(round((y_mm - self.grid.v0_mm) / pitch))
        return U, V

    def uv_to_mm(self, U: int, V: int) -> Tuple[float, float]:
        """Convert lattice grid coordinates to world coordinates"""
        assert self.grid is not None, "Grid not configured"
        pitch = self.grid.pitch_mm
        x_mm = self.grid.u0_mm + U * pitch
        y_mm = self.grid.v0_mm + V * pitch
        return x_mm, y_mm

    # ---- Authoritative Node Indexing ----

    def node_index(self, layer: int, u: int, v: int) -> int:
        """Single source of truth for node indexing"""
        assert self.grid is not None, "Grid not configured"
        return layer * (self.grid.u_steps * self.grid.v_steps) + v * self.grid.u_steps + u

    def node_index_to_luv(self, idx: int) -> Tuple[int, int, int]:
        """Single source of truth for node index → (layer, U, V) conversion"""
        assert self.grid is not None, "Grid not configured"
        u_steps, v_steps = self.grid.u_steps, self.grid.v_steps
        per_layer = u_steps * v_steps
        layer = idx // per_layer
        r = idx % per_layer
        v = r // u_steps
        u = r % u_steps
        # Tripwires
        assert 0 <= layer < len(self.layers), f"Layer {layer} out of range [0, {len(self.layers)})"
        assert 0 <= u < u_steps and 0 <= v < v_steps, f"UV ({u},{v}) out of bounds ({u_steps},{v_steps})"
        return layer, u, v

    @property
    def N(self) -> int:
        """Total number of nodes in the grid"""
        if not self.grid:
            return 0
        return len(self.layers) * self.grid.u_steps * self.grid.v_steps

    def node_idx_to_coord(self, node_idx: int, layers: int) -> Optional[Tuple[float, float, int]]:
        """Convert node index to (x, y, layer) coordinates using registry grid"""
        assert self.grid is not None, "Grid not configured"

        try:
            # Decode node index: node_idx = layer * (u_steps * v_steps) + v_idx * u_steps + u_idx
            u_steps = self.grid.u_steps
            v_steps = self.grid.v_steps
            layer_size = u_steps * v_steps

            layer = node_idx // layer_size
            remainder = node_idx % layer_size
            v_idx = remainder // u_steps
            u_idx = remainder % u_steps

            # Validate bounds
            if layer >= layers or u_idx >= u_steps or v_idx >= v_steps:
                logger.debug(f"[NODE-CONVERT] Bounds check failed: node_idx={node_idx}, "
                           f"layer={layer}/{layers}, u={u_idx}/{u_steps}, v={v_idx}/{v_steps}")
                return None

            # Convert to world coordinates
            x_mm, y_mm = self.uv_to_mm(u_idx, v_idx)
            return (float(x_mm), float(y_mm), int(layer))

        except Exception:
            return None

    def canon_key(self, layer_id: int, U1: int, V1: int, U2: int, V2: int) -> Key5:
        """Create canonical edge key with consistent ordering"""
        # Canonical ordering: (U2,V2) < (U1,V1) lexicographically
        if (V2, U2) < (V1, U1):
            U1, V1, U2, V2 = U2, V2, U1, V1
        return (layer_id, U1, V1, U2, V2)

    # ---- Edge Management ----

    def add_edge_uv(self, layer_id: int, U1: int, V1: int, U2: int, V2: int,
                    is_via: bool = False, base_cost: float = 1.0) -> int:
        """Add edge during lattice construction, returns edge ID"""
        assert not self._frozen, "Cannot add edges to frozen registry"

        key = self.canon_key(layer_id, U1, V1, U2, V2)

        # Return existing edge ID if already added
        if key in self._key5_to_eid:
            return self._key5_to_eid[key]

        # Create new edge
        eid = len(self._eid_to_key5)
        self._key5_to_eid[key] = eid
        self._eid_to_key5.append(key)

        # Track via edges
        if is_via:
            self._via_edge_ids.add(eid)

        return eid

    def enforce_hv_discipline(self, layer_id: int, U1: int, V1: int, U2: int, V2: int) -> bool:
        """Check if edge respects layer orientation rules"""
        if layer_id not in self.layers:
            return True  # Allow unknown layers

        layer_info = self.layers[layer_id]
        delta_u = abs(U2 - U1)
        delta_v = abs(V2 - V1)

        if layer_info.orientation == "H":
            # Horizontal layer: only allow horizontal edges (ΔU≠0, ΔV=0)
            return delta_u > 0 and delta_v == 0
        elif layer_info.orientation == "V":
            # Vertical layer: only allow vertical edges (ΔU=0, ΔV≠0)
            return delta_u == 0 and delta_v > 0
        elif layer_info.orientation == "VIA":
            # Via layer: allow single-point "edges" (for via tracking)
            return delta_u == 0 and delta_v == 0
        else:
            return True

    def add_edge_mm(self, layer_id: int, x1_mm: float, y1_mm: float,
                    x2_mm: float, y2_mm: float, is_via: bool = False,
                    base_cost: float = 1.0) -> Optional[int]:
        """Add edge using world coordinates, with H/V discipline enforcement"""
        U1, V1 = self.mm_to_uv(x1_mm, y1_mm)
        U2, V2 = self.mm_to_uv(x2_mm, y2_mm)

        # Enforce H/V discipline
        if not self.enforce_hv_discipline(layer_id, U1, V1, U2, V2):
            return None  # Reject edge that violates orientation rules

        return self.add_edge_uv(layer_id, U1, V1, U2, V2, is_via, base_cost)

    # ---- Net Management ----

    def get_or_create_net_idx(self, net_name: str) -> int:
        """Get dense index for net name, creating if needed"""
        if net_name not in self.net_to_idx:
            idx = len(self.idx_to_net)
            self.net_to_idx[net_name] = idx
            self.idx_to_net.append(net_name)
            self.fixed_eids_by_net[idx] = set()
        return self.net_to_idx[net_name]

    def get_net_name(self, net_idx: int) -> str:
        """Get net name from dense index"""
        assert 0 <= net_idx < len(self.idx_to_net), f"Invalid net index: {net_idx}"
        return self.idx_to_net[net_idx]

    def mark_edge_fixed(self, net_name: str, eid: int):
        """Mark edge as fixed (not rippable) for a net"""
        net_idx = self.get_or_create_net_idx(net_name)
        self.fixed_eids_by_net[net_idx].add(eid)

    def is_edge_fixed(self, net_idx: int, eid: int) -> bool:
        """Check if edge is fixed for a net"""
        return eid in self.fixed_eids_by_net.get(net_idx, set())

    # ---- Finalization ----


    def freeze(self):
        """Allocate dense arrays and freeze registry for routing"""
        if self._frozen:
            return

        self._frozen = True
        self.E = len(self._eid_to_key5)

        if self.E == 0:
            logger.warning("[REGISTRY] Freezing with 0 edges!")
            return

        # Allocate dense arrays
        self.eid_to_key5 = np.array(self._eid_to_key5, dtype=np.int32)
        self.base_cost = np.ones(self.E, dtype=np.float32)  # Default base cost = 1.0
        self.owner = np.full(self.E, -1, dtype=np.int32)
        self.hist_cost = np.zeros(self.E, dtype=np.float32)
        self.present_cost = np.zeros(self.E, dtype=np.float32)
        self.blocked = np.zeros(self.E, dtype=np.uint8)
        self.e_is_via = np.zeros(self.E, dtype=np.uint8)

        # Set via flags based on tracked via edges
        for eid in self._via_edge_ids:
            if eid < self.E:
                self.e_is_via[eid] = 1

        # NOTE: One-time clearance dilation reverted due to O(E²) performance

        logger.info(f"[REGISTRY] Frozen with {self.E:,} edges, {len(self.idx_to_net)} nets")

    # ---- Runtime Lookups ----

    def key_to_eid(self, key: Key5) -> int:
        """Convert canonical key to edge ID"""
        assert self._frozen, "Registry not frozen"
        return self._key5_to_eid[key]

    def eid_to_key(self, eid: int) -> Key5:
        """Convert edge ID to canonical key"""
        assert self._frozen, "Registry not frozen"
        assert 0 <= eid < self.E, f"Invalid edge ID: {eid}"
        return tuple(self.eid_to_key5[eid])

    def get_arrays(self) -> Tuple[np.ndarray, ...]:
        """Get all dense arrays for GPU/routing operations"""
        assert self._frozen, "Registry not frozen"
        return (self.owner, self.base_cost, self.hist_cost, self.present_cost,
                self.blocked, self.e_is_via, self.eid_to_key5)

    # ---- Validation ----

    def validate_integrity(self):
        """Validate internal consistency"""
        if not self._frozen:
            return

        assert self.E == len(self.eid_to_key5), f"Edge count mismatch: {self.E} vs {len(self.eid_to_key5)}"
        assert self.owner.shape[0] == self.E, f"Owner array size mismatch: {self.owner.shape[0]} vs {self.E}"
        assert len(self._key5_to_eid) == self.E, f"Key lookup size mismatch: {len(self._key5_to_eid)} vs {self.E}"

        # Validate key->eid->key roundtrip
        for eid in range(min(100, self.E)):  # Sample validation
            key = self.eid_to_key(eid)
            recovered_eid = self.key_to_eid(key)
            assert recovered_eid == eid, f"Roundtrip failed for eid={eid}: key={key}, recovered={recovered_eid}"

        logger.info(f"[REGISTRY] Validation passed: {self.E:,} edges, {len(self.layers)} layers")

# Global accessor
def get_registry() -> LatticeRegistry:
    """Get global geometry registry singleton"""
    return LatticeRegistry()

def reset_registry():
    """Reset registry for testing"""
    LatticeRegistry._instance = None