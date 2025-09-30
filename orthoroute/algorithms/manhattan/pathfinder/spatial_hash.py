"""
Spatial Hash Module

Simple spatial hash grid for fast DRC collision detection.
"""

from collections import defaultdict
from typing import Optional


class SpatialHash:
    """Simple spatial hash for fast DRC collision detection"""

    def __init__(self, cell_size: float):
        """Initialize spatial hash grid for collision detection.

        Args:
            cell_size: Size of each grid cell in mm for spatial partitioning
        """
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # cell_id -> list of segments

    def _hash_point(self, x: float, y: float) -> tuple:
        """Hash point to grid cell"""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_for_segment(self, p1: tuple, p2: tuple, radius: float) -> set:
        """Get all grid cells that a segment with radius might touch"""
        x1, y1 = p1
        x2, y2 = p2

        # Expand by radius
        min_x = min(x1, x2) - radius
        max_x = max(x1, x2) + radius
        min_y = min(y1, y2) - radius
        max_y = max(y1, y2) + radius

        # Get cell range
        min_cell_x = int(min_x // self.cell_size)
        max_cell_x = int(max_x // self.cell_size)
        min_cell_y = int(min_y // self.cell_size)
        max_cell_y = int(max_y // self.cell_size)

        cells = set()
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cells.add((cx, cy))
        return cells

    def insert_segment(self, p1: tuple, p2: tuple, radius: float, tag: str):
        """Insert segment into spatial hash"""
        cells = self._get_cells_for_segment(p1, p2, radius)
        segment_data = {'p1': p1, 'p2': p2, 'radius': radius, 'tag': tag}

        for cell in cells:
            self.grid[cell].append(segment_data)

    def query_segment(self, p1: tuple, p2: tuple, radius: float) -> list:
        """Query segments that might conflict with given segment"""
        cells = self._get_cells_for_segment(p1, p2, radius)
        candidates = []

        for cell in cells:
            for segment in self.grid.get(cell, []):
                # Simple distance check - in practice would use proper segment-segment distance
                candidates.append(type('Segment', (), {'tag': segment['tag']}))

        return candidates

    def nearest_distance(self, p1: tuple, p2: tuple, exclude_net: str, cap: float) -> Optional[float]:
        """Find nearest distance to other nets (simplified implementation)"""
        cells = self._get_cells_for_segment(p1, p2, cap)
        min_dist = None

        for cell in cells:
            for segment in self.grid.get(cell, []):
                if segment['tag'] != exclude_net:
                    # Simplified distance calculation
                    dist = ((p1[0] - segment['p1'][0])**2 + (p1[1] - segment['p1'][1])**2)**0.5
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

        return min_dist