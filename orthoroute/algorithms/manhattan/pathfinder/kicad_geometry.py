"""
KiCad Geometry Module

Single source of truth for all coordinate conversions based on KiCad board.
Handles lattice-to-world coordinate transformations and layer direction rules.
"""

from typing import Tuple
from .config import DEFAULT_GRID_PITCH


class KiCadGeometry:
    """Single source of truth for all coordinate conversions based on KiCad board"""

    def __init__(self, kicad_bounds: Tuple[float, float, float, float], pitch: float = DEFAULT_GRID_PITCH, layer_count: int = 2):
        """Initialize KiCad geometry system with bounds and grid pitch.

        Args:
            kicad_bounds: Tuple of (min_x, min_y, max_x, max_y) in mm
            pitch: Grid pitch in mm for routing lattice alignment
            layer_count: Number of copper layers from KiCad board (default: 2 for minimal boards)
        """
        self.min_x, self.min_y, self.max_x, self.max_y = kicad_bounds
        self.pitch = pitch

        # Grid aligned to pitch boundaries
        self.grid_min_x = round(self.min_x / pitch) * pitch
        self.grid_min_y = round(self.min_y / pitch) * pitch
        self.grid_max_x = round(self.max_x / pitch) * pitch
        self.grid_max_y = round(self.max_y / pitch) * pitch

        # Grid dimensions in lattice steps
        self.x_steps = int((self.grid_max_x - self.grid_min_x) / pitch) + 1
        self.y_steps = int((self.grid_max_y - self.grid_min_y) / pitch) + 1

        # Layer configuration - set from board.layer_count
        self.layer_count = layer_count
        self.layer_directions = ['h' if i % 2 == 0 else 'v' for i in range(layer_count)]

    def lattice_to_world(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        """Convert lattice indices to world coordinates"""
        world_x = self.grid_min_x + (x_idx * self.pitch)
        world_y = self.grid_min_y + (y_idx * self.pitch)
        return (world_x, world_y)

    def world_to_lattice(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to lattice indices"""
        x_idx = round((world_x - self.grid_min_x) / self.pitch)
        y_idx = round((world_y - self.grid_min_y) / self.pitch)
        return (x_idx, y_idx)

    def node_index(self, x_idx: int, y_idx: int, layer: int) -> int:
        """Convert lattice coordinates to flat node index"""
        layer_size = self.x_steps * self.y_steps
        return layer * layer_size + y_idx * self.x_steps + x_idx

    def index_to_coords(self, node_idx: int) -> Tuple[int, int, int]:
        """Convert flat node index back to lattice coordinates"""
        layer_size = self.x_steps * self.y_steps
        layer = node_idx // layer_size
        local_idx = node_idx % layer_size
        y_idx = local_idx // self.x_steps
        x_idx = local_idx % self.x_steps
        return (x_idx, y_idx, layer)

    def is_valid_edge(self, from_x: int, from_y: int, from_layer: int,
                      to_x: int, to_y: int, to_layer: int) -> bool:
        """Check if edge follows layer direction rules"""
        if from_layer != to_layer:
            return True  # Via connections always valid

        direction = self.layer_directions[from_layer]
        is_horizontal = (from_y == to_y and abs(from_x - to_x) == 1)
        is_vertical = (from_x == to_x and abs(from_y - to_y) == 1)

        if direction == 'h':
            return is_horizontal  # H-layers: only horizontal edges
        else:
            return is_vertical    # V-layers: only vertical edges