"""Domain models for PCB board structure."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4


@dataclass(frozen=True)
class Coordinate:
    """Value object representing a 2D coordinate in mm."""
    x: float
    y: float
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate Euclidean distance to another coordinate."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(frozen=True)
class Bounds:
    """Value object representing rectangular bounds."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Coordinate:
        return Coordinate(
            x=(self.min_x + self.max_x) / 2,
            y=(self.min_y + self.max_y) / 2
        )


@dataclass
class Pad:
    """Domain entity representing a component pad."""
    id: str
    component_id: str
    net_id: Optional[str]
    position: Coordinate
    size: Tuple[float, float]  # width, height
    drill_size: Optional[float] = None
    layer: str = "F.Cu"
    shape: str = "circle"
    angle: float = 0.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())


@dataclass
class Component:
    """Domain entity representing a PCB component."""
    id: str
    reference: str
    value: str
    footprint: str
    position: Coordinate
    angle: float = 0.0
    layer: str = "F.Cu"
    pads: List[Pad] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
        
        # Ensure all pads reference this component
        for pad in self.pads:
            pad.component_id = self.id
    
    def get_bounds(self) -> Bounds:
        """Calculate component bounds based on pad positions."""
        if not self.pads:
            return Bounds(self.position.x, self.position.y, 
                         self.position.x, self.position.y)
        
        min_x = min(pad.position.x - pad.size[0]/2 for pad in self.pads)
        min_y = min(pad.position.y - pad.size[1]/2 for pad in self.pads)
        max_x = max(pad.position.x + pad.size[0]/2 for pad in self.pads)
        max_y = max(pad.position.y + pad.size[1]/2 for pad in self.pads)
        
        return Bounds(min_x, min_y, max_x, max_y)


@dataclass
class Net:
    """Domain entity representing an electrical net."""
    id: str
    name: str
    netclass: str = "Default"
    pads: List[Pad] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
        
        # Update pad net references
        for pad in self.pads:
            pad.net_id = self.id
    
    @property
    def is_routable(self) -> bool:
        """Check if net has enough pads to require routing."""
        return len(self.pads) >= 2
    
    def get_bounds(self) -> Bounds:
        """Calculate net bounds based on pad positions."""
        if not self.pads:
            return Bounds(0, 0, 0, 0)
        
        min_x = min(pad.position.x for pad in self.pads)
        min_y = min(pad.position.y for pad in self.pads)
        max_x = max(pad.position.x for pad in self.pads)
        max_y = max(pad.position.y for pad in self.pads)
        
        return Bounds(min_x, min_y, max_x, max_y)
    
    def calculate_min_distance(self) -> float:
        """Calculate minimum distance between any two pads."""
        if len(self.pads) < 2:
            return 0.0
        
        min_dist = float('inf')
        for i in range(len(self.pads)):
            for j in range(i + 1, len(self.pads)):
                dist = self.pads[i].position.distance_to(self.pads[j].position)
                min_dist = min(min_dist, dist)
        
        return min_dist


@dataclass
class Layer:
    """Domain entity representing a PCB layer."""
    name: str
    type: str  # copper, signal, power, ground
    stackup_position: int
    thickness: float = 0.035  # mm, standard copper thickness
    material: str = "copper"
    
    @property
    def is_routing_layer(self) -> bool:
        """Check if layer can be used for routing."""
        return self.type in ['signal', 'power', 'ground'] and 'Cu' in self.name


@dataclass
class Board:
    """Domain aggregate root representing the PCB board."""
    id: str
    name: str
    components: List[Component] = field(default_factory=list)
    nets: List[Net] = field(default_factory=list)
    layers: List[Layer] = field(default_factory=list)
    
    # Board properties
    thickness: float = 1.6  # mm, standard PCB thickness
    layer_count: int = 2

    # Keepout rule areas from KiCad (each is a dict with 'outline', 'layers',
    # 'keepout_tracks', 'keepout_vias', 'keepout_copper', etc.)
    keepouts: List[Dict] = field(default_factory=list)
    
    # Mappings for efficient lookup
    _components_by_id: Dict[str, Component] = field(default_factory=dict, init=False)
    _nets_by_id: Dict[str, Net] = field(default_factory=dict, init=False)
    _nets_by_name: Dict[str, Net] = field(default_factory=dict, init=False)
    _layers_by_name: Dict[str, Layer] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
        
        self._build_indexes()
    
    def _build_indexes(self):
        """Build internal indexes for efficient lookup."""
        self._components_by_id = {comp.id: comp for comp in self.components}
        self._nets_by_id = {net.id: net for net in self.nets}
        self._nets_by_name = {net.name: net for net in self.nets}
        self._layers_by_name = {layer.name: layer for layer in self.layers}
    
    def add_component(self, component: Component) -> None:
        """Add a component to the board."""
        if component.id not in self._components_by_id:
            self.components.append(component)
            self._components_by_id[component.id] = component
    
    def add_net(self, net: Net) -> None:
        """Add a net to the board."""
        if net.id not in self._nets_by_id:
            self.nets.append(net)
            self._nets_by_id[net.id] = net
            self._nets_by_name[net.name] = net
    
    def add_layer(self, layer: Layer) -> None:
        """Add a layer to the board."""
        if layer.name not in self._layers_by_name:
            self.layers.append(layer)
            self._layers_by_name[layer.name] = layer
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get component by ID."""
        return self._components_by_id.get(component_id)
    
    def get_net(self, net_id: str) -> Optional[Net]:
        """Get net by ID."""
        return self._nets_by_id.get(net_id)
    
    def get_net_by_name(self, net_name: str) -> Optional[Net]:
        """Get net by name."""
        return self._nets_by_name.get(net_name)
    
    def get_layer(self, layer_name: str) -> Optional[Layer]:
        """Get layer by name."""
        return self._layers_by_name.get(layer_name)
    
    def get_routable_nets(self) -> List[Net]:
        """Get all nets that require routing (2+ pads)."""
        return [net for net in self.nets if net.is_routable]
    
    def get_routing_layers(self) -> List[Layer]:
        """Get all layers that can be used for routing."""
        return [layer for layer in self.layers if layer.is_routing_layer]
    
    def get_bounds(self) -> Bounds:
        """Calculate board bounds based on component positions."""
        if not self.components:
            return Bounds(0, 0, 0, 0)
        
        all_bounds = [comp.get_bounds() for comp in self.components]
        min_x = min(bounds.min_x for bounds in all_bounds)
        min_y = min(bounds.min_y for bounds in all_bounds)
        max_x = max(bounds.max_x for bounds in all_bounds)
        max_y = max(bounds.max_y for bounds in all_bounds)
        
        return Bounds(min_x, min_y, max_x, max_y)
    
    def get_all_pads(self) -> List[Pad]:
        """Get all pads from all components."""
        all_pads = []
        for component in self.components:
            all_pads.extend(component.pads)
        return all_pads
    
    def validate_integrity(self) -> List[str]:
        """Validate board integrity and return list of issues."""
        issues = []
        
        # Check for orphaned pads (pads with net_id but net doesn't exist)
        all_pads = self.get_all_pads()
        for pad in all_pads:
            if pad.net_id and pad.net_id not in self._nets_by_id:
                issues.append(f"Pad {pad.id} references non-existent net {pad.net_id}")
        
        # Check for nets with missing pads
        for net in self.nets:
            for pad in net.pads:
                if not any(p.id == pad.id for p in all_pads):
                    issues.append(f"Net {net.name} references non-existent pad {pad.id}")
        
        # Check for duplicate component references
        component_refs = [comp.reference for comp in self.components]
        duplicates = set([ref for ref in component_refs if component_refs.count(ref) > 1])
        for duplicate in duplicates:
            issues.append(f"Duplicate component reference: {duplicate}")
        
        return issues