"""
═══════════════════════════════════════════════════════════════════════════════
PAD ESCAPE PLANNER - PRECOMPUTED DRC-CLEAN ESCAPE ROUTING
═══════════════════════════════════════════════════════════════════════════════

This module handles precomputation of pad escape routing for multi-layer PCBs.
Before any pathfinding begins, we generate escape stubs and vias for all SMD
pads, distributing traffic across horizontal routing layers.

KEY FEATURES:
- Column-based processing: Group pads by x_idx, process entire columns atomically
- Distance-based direction: Choose direction based on nearest neighbor distance
- Inverted checkerboard fallback: (x + y) % 2 → even=DOWN, odd=UP (at board edges)
- Greedy pair-wise collision resolution: Alternately shorten escapes by 2 steps until no overlap
- Escape lengths: 3-12 grid steps (1.2mm - 4.8mm @ 0.4mm pitch), min 3 steps
- DRC checking against existing pads (0.15mm clearance)
- Vertical + 45-degree routing geometry

USAGE:
    planner = PadEscapePlanner(lattice, config, pad_to_node)
    tracks, vias = planner.precompute_all_pad_escapes(board)
"""

import logging
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .pathfinder.config import PAD_CLEARANCE_MM

logger = logging.getLogger(__name__)


@dataclass
class Portal:
    """Portal escape point for a pad"""
    x_idx: int          # Lattice x-coordinate of portal
    y_idx: int          # Lattice y-coordinate of portal (offset from pad)
    pad_layer: int      # Physical pad layer (e.g., F.Cu = 0)
    delta_steps: int    # Vertical offset from pad (3-12 steps)
    direction: int      # +1 (up) or -1 (down)
    pad_x: float        # Original pad x in mm
    pad_y: float        # Original pad y in mm
    score: float = 0.0  # Quality score (lower is better)
    retarget_count: int = 0  # How many times retargeted


class PadEscapePlanner:
    """Plans and generates DRC-clean escape routing for SMD pads"""

    def __init__(self, lattice, config, pad_to_node: Dict):
        """
        Initialize pad escape planner.

        Args:
            lattice: Lattice3D instance with grid geometry
            config: PathFinderConfig with routing parameters
            pad_to_node: Dict[pad_id -> node_idx] mapping
        """
        self.lattice = lattice
        self.config = config
        self.pad_to_node = pad_to_node
        self.portals: Dict[str, Portal] = {}  # pad_id -> Portal

    def precompute_all_pad_escapes(self, board, nets_to_route: List = None) -> Tuple[List, List]:
        """
        Precompute escape routing for SMD pads using column-based processing.

        Algorithm:
        1. Collect all routable pads and snap to grid columns
        2. Group pads by column (x_idx)
        3. For each column, sort pads by y_idx
        4. Determine direction for each pad:
           - Find nearest neighbor above/below in same column
           - Choose direction with most distance
           - At board edges (no neighbor), use inverted checkerboard
        5. Start all escapes at max length (12 steps)
        6. Greedy collision resolution:
           - Check pairs of escapes in column for Y-range overlap
           - If collision, alternately shorten by 2 steps until resolved
           - Min length = 3 steps
        7. DRC check and emit geometry

        Args:
            board: Board with components and pads
            nets_to_route: List of net names to route (if None, uses board.nets)

        Returns (tracks, vias) for visualization.
        """
        tracks = []
        vias = []

        # Use existing pad geometries from board_data (already extracted by rich_kicad_interface)
        raw_pads = getattr(board, '_gui_pads', [])
        if not raw_pads:
            logger.warning("No GUI pads found on board, using fallback extraction")
            pad_geometries = self._extract_pad_geometries(board)
        else:
            # Build pad_id -> geometry mapping from GUI pads
            pad_geometries = {}
            for pad_dict in raw_pads:
                x, y = pad_dict['x'], pad_dict['y']

                # Find pad_id by matching position (within tolerance)
                for pid in self.pad_to_node.keys():
                    if '@' in pid:
                        try:
                            coords_str = pid.split('@')[1]
                            px_microns, py_microns = map(int, coords_str.split(','))
                            px_mm = px_microns / 1000.0
                            py_mm = py_microns / 1000.0

                            # Match if within 0.01mm
                            if abs(px_mm - x) < 0.01 and abs(py_mm - y) < 0.01:
                                pad_geometries[pid] = {
                                    'x': x,
                                    'y': y,
                                    'width': pad_dict['width'],
                                    'height': pad_dict['height']
                                }
                                break
                        except:
                            continue

            logger.info(f"Mapped {len(pad_geometries)} pad geometries from GUI data")

        # Parse nets from board directly
        if nets_to_route is None:
            nets_to_route = [net for net in getattr(board, 'nets', [])]

        logger.info(f"Planning escapes for {len(nets_to_route)} nets")

        # Build set of routable pad IDs by examining nets directly
        routable_pad_ids = set()
        net_pad_mapping = {}  # net_name -> (pad_id1, pad_id2)

        for net in nets_to_route:
            if not hasattr(net, 'name') or not hasattr(net, 'pads'):
                continue

            net_name = net.name
            pads = net.pads

            if len(pads) < 2:
                continue

            # Get pad IDs for first two pads in net
            p1, p2 = pads[0], pads[1]
            p1_id = self._pad_key(p1)
            p2_id = self._pad_key(p2)

            # Only include pads that are actually mapped
            if p1_id in self.pad_to_node and p2_id in self.pad_to_node:
                routable_pad_ids.add(p1_id)
                routable_pad_ids.add(p2_id)
                net_pad_mapping[net_name] = (p1_id, p2_id)

        logger.info(f"Found {len(routable_pad_ids)} pads attached to {len(net_pad_mapping)} routable nets")

        # Clear existing portals
        self.portals.clear()

        # STEP 1: Collect all routable pads with grid positions
        pad_list = []  # [(pad_obj, pad_id, x_idx, y_idx, pad_x, pad_y, pad_layer), ...]

        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                # Skip through-hole pads
                drill = getattr(pad, 'drill', 0.0)
                if drill > 0:
                    continue

                pad_id = self._pad_key(pad, comp)
                if pad_id not in routable_pad_ids:
                    continue

                pad_x, pad_y = pad.position.x, pad.position.y
                pad_layer = self._get_pad_layer(pad)

                # Snap to grid
                x_idx_nearest, y_idx_nearest = self.lattice.world_to_lattice(pad_x, pad_y)
                x_idx_nearest = max(0, min(x_idx_nearest, self.lattice.x_steps - 1))
                y_idx_nearest = max(0, min(y_idx_nearest, self.lattice.y_steps - 1))

                # Check snap tolerance
                x_mm_snapped, _ = self.lattice.geom.lattice_to_world(x_idx_nearest, 0)
                x_snap_dist_steps = abs(pad_x - x_mm_snapped) / self.config.grid_pitch

                if x_snap_dist_steps > self.config.portal_x_snap_max:
                    logger.debug(f"Pad {pad_id}: x-snap {x_snap_dist_steps:.2f} exceeds max")
                    continue

                pad_list.append((pad, pad_id, x_idx_nearest, y_idx_nearest, pad_x, pad_y, pad_layer))

        # Board-level pads
        for pad in getattr(board, "pads", []):
            drill = getattr(pad, 'drill', 0.0)
            if drill > 0:
                continue

            pad_id = self._pad_key(pad, comp=None)
            if pad_id not in routable_pad_ids:
                continue

            pad_x, pad_y = pad.position.x, pad.position.y
            pad_layer = self._get_pad_layer(pad)

            # Snap to grid
            x_idx_nearest, y_idx_nearest = self.lattice.world_to_lattice(pad_x, pad_y)
            x_idx_nearest = max(0, min(x_idx_nearest, self.lattice.x_steps - 1))
            y_idx_nearest = max(0, min(y_idx_nearest, self.lattice.y_steps - 1))

            # Check snap tolerance
            x_mm_snapped, _ = self.lattice.geom.lattice_to_world(x_idx_nearest, 0)
            x_snap_dist_steps = abs(pad_x - x_mm_snapped) / self.config.grid_pitch

            if x_snap_dist_steps > self.config.portal_x_snap_max:
                logger.debug(f"Pad {pad_id}: x-snap {x_snap_dist_steps:.2f} exceeds max")
                continue

            pad_list.append((pad, pad_id, x_idx_nearest, y_idx_nearest, pad_x, pad_y, pad_layer))

        logger.info(f"Collected {len(pad_list)} pads for escape planning")

        # STEP 2: Group by column (x_idx)
        columns = {}  # x_idx -> [(pad_obj, pad_id, y_idx, pad_x, pad_y, pad_layer), ...]
        for pad, pad_id, x_idx, y_idx, pad_x, pad_y, pad_layer in pad_list:
            if x_idx not in columns:
                columns[x_idx] = []
            columns[x_idx].append((pad, pad_id, y_idx, pad_x, pad_y, pad_layer))

        # STEP 3: Process each column
        portal_count = 0
        for x_idx, column_pads in columns.items():
            # Sort by y_idx
            column_pads.sort(key=lambda p: p[2])  # Sort by y_idx

            # Plan portals for this column
            portals_created = self._plan_column_escapes(x_idx, column_pads, pad_geometries)
            portal_count += portals_created

        logger.info(f"Planned {portal_count} portals using column-based approach")

        # Build reverse lookup: pad_id -> net_id
        pad_to_net = {}
        for net_id, (src_pad_id, dst_pad_id) in net_pad_mapping.items():
            pad_to_net[src_pad_id] = net_id
            pad_to_net[dst_pad_id] = net_id

        # FIRST PASS: Generate all escape geometry
        portal_geometry = {}  # pad_id -> (tracks, vias, portal, entry_layer)
        for pad_id, portal in self.portals.items():
            net_id = pad_to_net.get(pad_id, f"PAD_{pad_id}")

            # Pick a random horizontal layer (odd indices)
            odd_layers = [i for i in range(1, self.lattice.layers, 2)]
            if not odd_layers:
                odd_layers = [1]
            entry_layer = random.choice(odd_layers)

            # Generate escape geometry (stub + via)
            geometry = self._emit_portal_escape_geometry(net_id, pad_id, portal, entry_layer)

            portal_tracks = []
            portal_vias = []
            for item in geometry:
                if 'x1' in item and 'y1' in item:  # It's a track
                    portal_tracks.append(item)
                elif 'x' in item and 'y' in item:  # It's a via
                    portal_vias.append(item)

            portal_geometry[pad_id] = (portal_tracks, portal_vias, portal, entry_layer)

        logger.info(f"Generated {len(portal_geometry)} escape geometries")

        # Collect all final geometry
        for pad_id, (portal_tracks, portal_vias, portal, entry_layer) in portal_geometry.items():
            tracks.extend(portal_tracks)
            vias.extend(portal_vias)

        logger.info(f"Final: {len(tracks)} escape stubs and {len(vias)} portal vias")
        return (tracks, vias)

    def _plan_column_escapes(self, x_idx: int, column_pads: List, pad_geometries: Dict) -> int:
        """
        Plan escapes for all pads in a single column using greedy collision resolution.

        Args:
            x_idx: Column index
            column_pads: List of (pad_obj, pad_id, y_idx, pad_x, pad_y, pad_layer) sorted by y_idx
            pad_geometries: Dict of pad geometries for DRC

        Returns:
            Number of portals successfully created
        """
        min_steps = 3
        max_steps = 12

        # STEP 1: Determine direction for each pad
        # [(pad_id, y_idx, direction, delta_steps, pad_x, pad_y, pad_layer), ...]
        planned_escapes = []

        for i, (pad, pad_id, y_idx, pad_x, pad_y, pad_layer) in enumerate(column_pads):
            # Find nearest neighbor above and below
            dist_above = None
            dist_below = None

            if i > 0:
                _, _, y_below, _, _, _ = column_pads[i - 1]
                dist_below = y_idx - y_below

            if i < len(column_pads) - 1:
                _, _, y_above, _, _, _ = column_pads[i + 1]
                dist_above = y_above - y_idx

            # Choose direction based on distance
            if dist_above is not None and dist_below is not None:
                # Both neighbors exist - choose direction with more space
                direction = +1 if dist_above > dist_below else -1
            elif dist_above is not None:
                # Only neighbor above - go up
                direction = +1
            elif dist_below is not None:
                # Only neighbor below - go down
                direction = -1
            else:
                # No neighbors - use inverted checkerboard (even=DOWN, odd=UP)
                checkerboard_value = (x_idx + y_idx) % 2
                direction = -1 if checkerboard_value == 0 else +1

            # Calculate max possible length based on board bounds
            if direction > 0:
                max_possible = self.lattice.y_steps - 1 - y_idx
            else:
                max_possible = y_idx

            # Calculate available space based on neighbor in escape direction
            # This prevents regular horizontal lines by constraining randomness to local density
            if direction > 0 and dist_above is not None:
                # Going up - limit based on distance to pad above
                # Leave buffer space (use half the gap)
                available_space = dist_above // 2
            elif direction < 0 and dist_below is not None:
                # Going down - limit based on distance to pad below
                available_space = dist_below // 2
            else:
                # No neighbor in escape direction - use full range
                available_space = max_steps

            # Combine all constraints
            safe_max = min(max_steps, available_space, max_possible)
            safe_max = max(safe_max, min_steps)  # Ensure at least min_steps

            # Pick random length within safe range for variety
            delta_steps = random.randint(min_steps, safe_max)

            planned_escapes.append((pad_id, y_idx, direction, delta_steps, pad_x, pad_y, pad_layer))

        # STEP 2: Greedy collision resolution - check all pairs and shorten as needed
        max_iterations = 100
        for iteration in range(max_iterations):
            collision_found = False

            # Check all pairs for Y-range overlap
            for i in range(len(planned_escapes)):
                for j in range(i + 1, len(planned_escapes)):
                    pad_id_a, y_idx_a, dir_a, delta_a, pad_x_a, pad_y_a, layer_a = planned_escapes[i]
                    pad_id_b, y_idx_b, dir_b, delta_b, pad_x_b, pad_y_b, layer_b = planned_escapes[j]

                    # Calculate Y-ranges
                    y_portal_a = y_idx_a + dir_a * delta_a
                    y_min_a = min(y_idx_a, y_portal_a)
                    y_max_a = max(y_idx_a, y_portal_a)

                    y_portal_b = y_idx_b + dir_b * delta_b
                    y_min_b = min(y_idx_b, y_portal_b)
                    y_max_b = max(y_idx_b, y_portal_b)

                    # Check overlap (with 1-step buffer)
                    if not (y_max_a + 1 < y_min_b or y_max_b + 1 < y_min_a):
                        # Collision! Shorten one alternately
                        collision_found = True

                        # Alternate which one to shorten based on iteration
                        if iteration % 2 == 0:
                            # Shorten A
                            if delta_a > min_steps:
                                new_delta_a = max(min_steps, delta_a - 2)
                                planned_escapes[i] = (pad_id_a, y_idx_a, dir_a, new_delta_a, pad_x_a, pad_y_a, layer_a)
                        else:
                            # Shorten B
                            if delta_b > min_steps:
                                new_delta_b = max(min_steps, delta_b - 2)
                                planned_escapes[j] = (pad_id_b, y_idx_b, dir_b, new_delta_b, pad_x_b, pad_y_b, layer_b)

            if not collision_found:
                break

        # STEP 3: DRC check and create portals with fallback
        portal_count = 0
        for pad_id, y_idx, direction, delta_steps, pad_x, pad_y, pad_layer in planned_escapes:
            # Try progressively shorter lengths, then opposite direction
            portal = None

            # First try: current direction, progressively shorter
            for try_delta in range(delta_steps, min_steps - 1, -1):
                portal = self._try_create_portal(x_idx, y_idx, direction, try_delta,
                                                  pad_id, pad_x, pad_y, pad_layer, pad_geometries)
                if portal:
                    break

            # Second try: opposite direction if first failed
            if not portal:
                opposite_direction = -direction
                # Calculate max possible in opposite direction
                if opposite_direction > 0:
                    max_opposite = self.lattice.y_steps - 1 - y_idx
                else:
                    max_opposite = y_idx
                max_opposite = min(max_steps, max_opposite)

                # Try opposite direction from min to max
                for try_delta in range(min_steps, max_opposite + 1):
                    portal = self._try_create_portal(x_idx, y_idx, opposite_direction, try_delta,
                                                      pad_id, pad_x, pad_y, pad_layer, pad_geometries)
                    if portal:
                        break

            if portal:
                self.portals[pad_id] = portal
                portal_count += 1
            else:
                logger.warning(f"Pad {pad_id}: could not find valid escape in any direction")

        return portal_count

    def _try_create_portal(self, x_idx: int, y_idx: int, direction: int, delta_steps: int,
                           pad_id: str, pad_x: float, pad_y: float, pad_layer: int,
                           pad_geometries: Dict) -> Optional[Portal]:
        """
        Try to create a portal with given parameters, return None if DRC fails.
        Uses LOCAL DRC checking (only nearby pads).
        """
        y_idx_portal = y_idx + direction * delta_steps

        # Convert portal to world coordinates
        portal_x_mm, portal_y_mm = self.lattice.geom.lattice_to_world(x_idx, y_idx_portal)

        # DRC check: portal via clearance (LOCAL only - within 3mm radius)
        if not self._check_clearance_to_pads(portal_x_mm, portal_y_mm, pad_id, pad_geometries,
                                              check_radius=3.0):
            return None

        # DRC check: stub path (LOCAL only)
        for t in [0.25, 0.5, 0.75]:
            stub_x = pad_x + t * (portal_x_mm - pad_x)
            stub_y = pad_y + t * (portal_y_mm - pad_y)
            if not self._check_clearance_to_pads(stub_x, stub_y, pad_id, pad_geometries,
                                                   check_radius=3.0):
                return None

        # DRC passed! Create portal
        return Portal(
            x_idx=x_idx,
            y_idx=y_idx_portal,
            pad_layer=pad_layer,
            delta_steps=delta_steps,
            direction=direction,
            pad_x=pad_x,
            pad_y=pad_y,
            score=0.0,
            retarget_count=0
        )

    def _pad_key(self, pad, comp=None):
        """Generate unique pad key with coordinates for orphaned pads"""
        comp_id = getattr(pad, "component_id", None) or (getattr(comp, "id", None) if comp else None) or "GENERIC_COMPONENT"

        # For orphaned pads, include coordinates to ensure uniqueness
        if comp_id == "GENERIC_COMPONENT" and hasattr(pad, 'position'):
            xq = int(round(pad.position.x * 1000))
            yq = int(round(pad.position.y * 1000))
            return f"{comp_id}_{pad.id}@{xq},{yq}"

        return f"{comp_id}_{pad.id}"

    def _get_pad_layer(self, pad) -> int:
        """Get the layer index for a pad with fallback handling"""
        # For now, all SMD pads default to F.Cu (layer 0)
        return 0

    def _extract_pad_geometries(self, board) -> Dict:
        """
        Extract geometry (position, size) for all pads for DRC checking.
        Fallback method if GUI pads not available.

        Returns dict: pad_id -> {x, y, width, height}
        """
        geometries = {}

        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                pad_id = self._pad_key(pad, comp)
                x = pad.position.x
                y = pad.position.y

                if hasattr(pad, 'size'):
                    size_x_iu = pad.size.x if hasattr(pad.size, 'x') else pad.size[0]
                    size_y_iu = pad.size.y if hasattr(pad.size, 'y') else pad.size[1]
                    width = size_x_iu / 1_000_000.0
                    height = size_y_iu / 1_000_000.0
                else:
                    width = 0.5
                    height = 0.5
                    logger.warning(f"Pad {pad_id}: no size attribute, using default 0.5mm")

                geometries[pad_id] = {'x': x, 'y': y, 'width': width, 'height': height}

        for pad in getattr(board, "pads", []):
            pad_id = self._pad_key(pad, comp=None)
            if pad_id not in geometries:
                x = pad.position.x
                y = pad.position.y

                if hasattr(pad, 'size'):
                    size_x_iu = pad.size.x if hasattr(pad.size, 'x') else pad.size[0]
                    size_y_iu = pad.size.y if hasattr(pad.size, 'y') else pad.size[1]
                    width = size_x_iu / 1_000_000.0
                    height = size_y_iu / 1_000_000.0
                else:
                    width = 0.5
                    height = 0.5

                geometries[pad_id] = {'x': x, 'y': y, 'width': width, 'height': height}

        return geometries

    def _check_clearance_to_pads(self, x: float, y: float, current_pad_id: str,
                                  pad_geometries: Dict, clearance_mm: float = None,
                                  debug: bool = False, check_radius: float = None) -> bool:
        """
        Check if point (x, y) maintains clearance_mm from other pads.

        Args:
            check_radius: If specified, only check pads within this radius (mm) for performance.
                         If None, check all pads (slower but thorough).

        Returns True if clearance is OK, False if violation.
        """
        if clearance_mm is None:
            clearance_mm = PAD_CLEARANCE_MM

        violations = []

        for pad_id, geom in pad_geometries.items():
            if pad_id == current_pad_id:
                continue  # Skip self

            pad_x = geom['x']
            pad_y = geom['y']

            # Quick radius check for performance
            if check_radius is not None:
                dx = abs(x - pad_x)
                dy = abs(y - pad_y)
                if dx > check_radius or dy > check_radius:
                    continue  # Skip distant pads

            # Calculate distance from point to pad bounding box
            pad_w = geom['width']
            pad_h = geom['height']

            # Expand pad by clearance to create keepout zone
            keepout_x_min = pad_x - pad_w / 2.0 - clearance_mm
            keepout_x_max = pad_x + pad_w / 2.0 + clearance_mm
            keepout_y_min = pad_y - pad_h / 2.0 - clearance_mm
            keepout_y_max = pad_y + pad_h / 2.0 + clearance_mm

            # Check if point is inside keepout zone
            if (keepout_x_min <= x <= keepout_x_max and
                keepout_y_min <= y <= keepout_y_max):
                if debug:
                    # Calculate actual distance
                    dx = abs(x - pad_x) - pad_w / 2.0
                    dy = abs(y - pad_y) - pad_h / 2.0
                    dist = max(dx, dy)
                    violations.append((pad_id, dist, geom))
                else:
                    return False  # Violation!

        if debug and violations:
            logger.info(f"  Point ({x:.2f}, {y:.2f}) violations:")
            for vid, dist, geom in violations[:3]:
                logger.info(f"    - Near {vid}: dist={dist:.3f}mm, pad_size=({geom['width']:.3f}×{geom['height']:.3f})")
            return False

        return len(violations) == 0

    def _emit_portal_escape_geometry(self, net_id: str, pad_id: str, portal: Portal, entry_layer: int):
        """Emit vertical + 45-degree escape stub and portal via for a pad"""
        geometry = []

        # 1. Escape routing: vertical segment + 45-degree segment to portal via
        pad_layer_name = self.config.layer_names[portal.pad_layer] if portal.pad_layer < len(self.config.layer_names) else f"L{portal.pad_layer}"

        # Get portal mm coordinates
        portal_x_mm, portal_y_mm = self.lattice.geom.lattice_to_world(portal.x_idx, portal.y_idx)

        # Calculate escape geometry: mostly vertical, then 45-degree to via
        dx = portal_x_mm - portal.pad_x
        dy = portal_y_mm - portal.pad_y

        # For 45-degree segment, we need dx == dy_45
        # If there's any horizontal offset, we'll use a 45-degree segment at the end
        if abs(dx) > 0.01:  # More than 0.01mm horizontal offset
            # Intermediate point: vertical from pad, then 45-degree to via
            # The 45-degree segment covers |dx| in both X and Y
            sign_y = 1 if dy > 0 else -1
            dy_45 = sign_y * abs(dx)  # 45-degree segment Y component (same magnitude as dx)
            dy_vertical = dy - dy_45   # Remaining Y distance covered by vertical segment

            # Intermediate point at end of vertical segment
            intermediate_x = portal.pad_x
            intermediate_y = portal.pad_y + dy_vertical

            # Vertical segment from pad to intermediate point
            if abs(dy_vertical) > 0.01:  # Only create segment if length > 0.01mm
                geometry.append({
                    'net': net_id,
                    'layer': pad_layer_name,
                    'x1': portal.pad_x,
                    'y1': portal.pad_y,
                    'x2': intermediate_x,
                    'y2': intermediate_y,
                    'width': self.config.grid_pitch * 0.6,
                })

            # 45-degree segment from intermediate point to portal via
            geometry.append({
                'net': net_id,
                'layer': pad_layer_name,
                'x1': intermediate_x,
                'y1': intermediate_y,
                'x2': portal_x_mm,
                'y2': portal_y_mm,
                'width': self.config.grid_pitch * 0.6,
            })
        else:
            # Pure vertical escape (no horizontal offset)
            geometry.append({
                'net': net_id,
                'layer': pad_layer_name,
                'x1': portal.pad_x,
                'y1': portal.pad_y,
                'x2': portal_x_mm,
                'y2': portal_y_mm,
                'width': self.config.grid_pitch * 0.6,
            })

        # 2. Portal via stack (minimal: only from pad_layer to entry_layer)
        if portal.pad_layer != entry_layer:
            entry_layer_name = self.config.layer_names[entry_layer] if entry_layer < len(self.config.layer_names) else f"L{entry_layer}"

            geometry.append({
                'net': net_id,
                'x': portal_x_mm,
                'y': portal_y_mm,
                'from_layer': pad_layer_name,
                'to_layer': entry_layer_name,
                'diameter': 0.25,  # hole (0.15) + 2×annular (0.05) = 0.25mm
                'drill': 0.15,     # hole diameter
            })

        return geometry
