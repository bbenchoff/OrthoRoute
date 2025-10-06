"""
═══════════════════════════════════════════════════════════════════════════════
PAD ESCAPE PLANNER - PRECOMPUTED DRC-CLEAN ESCAPE ROUTING
═══════════════════════════════════════════════════════════════════════════════

This module handles precomputation of pad escape routing for multi-layer PCBs.
Before any pathfinding begins, we generate escape stubs and vias for all SMD
pads, distributing traffic across horizontal routing layers.

KEY FEATURES:
- Random escape lengths (1.2mm - 5mm) and directions (±Y)
- DRC checking against existing pads (0.15mm clearance)
- Via-to-via conflict resolution (0.4mm clearance)
- Track-to-via conflict resolution (0.275mm clearance)
- Iterative retry logic for conflicting escapes

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
        Precompute escape routing for SMD pads attached to nets we want to route.

        For each SMD pad on a routable net:
        1. Snap X to nearest grid column (±½ pitch allowed)
        2. Pick random vertical length d ∈ {3..12} grid steps (1.2mm - 4.8mm @ 0.4mm pitch)
        3. Pick random direction (EITHER up OR down), clamped to board bounds
        4. DRC check: ensure stub and via maintain clearance from other pads
        5. Compute stub tip (xg, yg±d) on F.Cu
        6. Place via to random horizontal layer (odd index: In1, In3, ..., B.Cu)

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

        # Debug: log sample pad geometries
        sample_pads = list(pad_geometries.items())[:5]
        for pad_id, geom in sample_pads:
            logger.info(f"  Sample pad {pad_id}: pos=({geom['x']:.3f}, {geom['y']:.3f}), size=({geom['width']:.3f} × {geom['height']:.3f})")

        # Parse nets from board directly (since net_pad_ids isn't populated yet)
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

        # Clear existing portals and plan ONLY for routable pads
        self.portals.clear()

        # Plan portals only for routable pads
        portal_count = 0
        drc_failures_logged = 0
        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                # Skip through-hole pads
                drill = getattr(pad, 'drill', 0.0)
                if drill > 0:
                    continue

                pad_id = self._pad_key(pad, comp)
                if pad_id not in routable_pad_ids:
                    continue

                portal = self._plan_random_portal_for_pad_with_drc(pad, pad_id, pad_geometries,
                                                                     debug=(drc_failures_logged < 3))
                if portal:
                    self.portals[pad_id] = portal
                    portal_count += 1
                else:
                    drc_failures_logged += 1

        # Board-level pads
        for pad in getattr(board, "pads", []):
            drill = getattr(pad, 'drill', 0.0)
            if drill > 0:
                continue

            pad_id = self._pad_key(pad, comp=None)
            if pad_id not in routable_pad_ids or pad_id in self.portals:
                continue

            portal = self._plan_random_portal_for_pad_with_drc(pad, pad_id, pad_geometries,
                                                                 debug=(drc_failures_logged < 3))
            if portal:
                self.portals[pad_id] = portal
                portal_count += 1
            else:
                drc_failures_logged += 1

        logger.info(f"Planned {portal_count} portals for routable nets (using RANDOM direction and 1.2-5mm length)")

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

        logger.info(f"First pass: generated {len(portal_geometry)} escape geometries")

        # SECOND PASS: Check for conflicts and retry
        max_retries = 3
        for retry_iteration in range(max_retries):
            conflicts = self._check_escape_conflicts(portal_geometry, pad_geometries)

            if not conflicts:
                logger.info(f"Second pass (iteration {retry_iteration + 1}): No conflicts detected!")
                break

            logger.info(f"Second pass (iteration {retry_iteration + 1}): Found {len(conflicts)} conflicts, regenerating...")

            # Retry conflicting portals
            for pad_id in conflicts:
                # Get the pad object to regenerate portal
                pad_obj = None
                for comp in getattr(board, "components", []):
                    for pad in getattr(comp, "pads", []):
                        if self._pad_key(pad, comp) == pad_id:
                            pad_obj = pad
                            break
                    if pad_obj:
                        break

                if not pad_obj:
                    for pad in getattr(board, "pads", []):
                        if self._pad_key(pad, comp=None) == pad_id:
                            pad_obj = pad
                            break

                if not pad_obj:
                    logger.warning(f"Could not find pad object for {pad_id}, skipping retry")
                    continue

                # Regenerate portal
                new_portal = self._plan_random_portal_for_pad_with_drc(pad_obj, pad_id, pad_geometries, debug=False)

                if new_portal:
                    net_id = pad_to_net.get(pad_id, f"PAD_{pad_id}")
                    odd_layers = [i for i in range(1, self.lattice.layers, 2)]
                    if not odd_layers:
                        odd_layers = [1]
                    entry_layer = random.choice(odd_layers)

                    geometry = self._emit_portal_escape_geometry(net_id, pad_id, new_portal, entry_layer)

                    portal_tracks = []
                    portal_vias = []
                    for item in geometry:
                        if 'x1' in item and 'y1' in item:
                            portal_tracks.append(item)
                        elif 'x' in item and 'y' in item:
                            portal_vias.append(item)

                    portal_geometry[pad_id] = (portal_tracks, portal_vias, new_portal, entry_layer)
                    logger.debug(f"Regenerated escape for {pad_id}")

        # Collect all final geometry
        for pad_id, (portal_tracks, portal_vias, portal, entry_layer) in portal_geometry.items():
            tracks.extend(portal_tracks)
            vias.extend(portal_vias)

        logger.info(f"Final: {len(tracks)} escape stubs and {len(vias)} portal vias")
        return (tracks, vias)

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
                                  debug: bool = False) -> bool:
        """
        Check if point (x, y) maintains clearance_mm from all other pads.

        Returns True if clearance is OK, False if violation.
        """
        if clearance_mm is None:
            clearance_mm = PAD_CLEARANCE_MM

        violations = []

        for pad_id, geom in pad_geometries.items():
            if pad_id == current_pad_id:
                continue  # Skip self

            # Calculate distance from point to pad bounding box
            pad_x = geom['x']
            pad_y = geom['y']
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

    def _plan_random_portal_for_pad_with_drc(self, pad, pad_id: str,
                                              pad_geometries: Dict, debug: bool = False) -> Optional[Portal]:
        """
        Plan portal escape with RANDOM direction and offset, WITH DRC checking.

        Length: 1.2mm - 5mm (3-12 grid steps @ 0.4mm pitch)
        Direction: Pick EITHER +1 (up) or -1 (down) randomly
        DRC: Ensure clearance from all other pads
        """
        # Get pad position and layer
        pad_x, pad_y = pad.position.x, pad.position.y
        pad_layer = self._get_pad_layer(pad)

        # Snap pad x to nearest lattice column
        x_idx_nearest, _ = self.lattice.world_to_lattice(pad_x, pad_y)
        x_idx_nearest = max(0, min(x_idx_nearest, self.lattice.x_steps - 1))

        # Check if snap is within tolerance
        x_mm_snapped, _ = self.lattice.geom.lattice_to_world(x_idx_nearest, 0)
        x_snap_dist_steps = abs(pad_x - x_mm_snapped) / self.config.grid_pitch

        if x_snap_dist_steps > self.config.portal_x_snap_max:
            logger.debug(f"Pad {pad_id}: x-snap {x_snap_dist_steps:.2f} exceeds max")
            return None

        x_idx = x_idx_nearest

        # Get pad y index
        _, y_idx_pad = self.lattice.world_to_lattice(pad_x, pad_y)
        y_idx_pad = max(0, min(y_idx_pad, self.lattice.y_steps - 1))

        # STEP 1: Pick direction FIRST (up or down)
        direction = random.choice([+1, -1])

        # STEP 2: Find valid length range in chosen direction
        min_steps = 3
        max_steps = 12

        # Calculate max possible steps in chosen direction before hitting board edge
        if direction > 0:
            max_possible = self.lattice.y_steps - 1 - y_idx_pad
        else:
            max_possible = y_idx_pad

        # Clamp max_steps to what's physically possible
        max_steps_bounded = min(max_steps, max_possible)

        if max_steps_bounded < min_steps:
            # Not enough space in this direction, try opposite
            direction = -direction
            if direction > 0:
                max_possible = self.lattice.y_steps - 1 - y_idx_pad
            else:
                max_possible = y_idx_pad
            max_steps_bounded = min(max_steps, max_possible)

            if max_steps_bounded < min_steps:
                logger.debug(f"Pad {pad_id}: insufficient space in either direction")
                return None

        # STEP 3: Try multiple random lengths in the chosen direction with DRC
        max_attempts = 10
        for attempt in range(max_attempts):
            # Random length within valid range for chosen direction
            delta_steps = random.randint(min_steps, max_steps_bounded)

            # Calculate portal position in chosen direction
            y_idx_portal = y_idx_pad + direction * delta_steps

            # Convert portal to world coordinates for DRC check
            portal_x_mm, portal_y_mm = self.lattice.geom.lattice_to_world(x_idx, y_idx_portal)

            # DRC check: verify portal via position maintains clearance
            if not self._check_clearance_to_pads(portal_x_mm, portal_y_mm, pad_id, pad_geometries):
                if attempt == 0 and debug:
                    logger.info(f"Pad {pad_id}: portal at ({portal_x_mm:.2f}, {portal_y_mm:.2f}) violates clearance:")
                    self._check_clearance_to_pads(portal_x_mm, portal_y_mm, pad_id,
                                                   pad_geometries, debug=True)
                continue

            # DRC check: verify stub path
            stub_clear = True
            for t in [0.25, 0.5, 0.75]:
                stub_x = pad_x + t * (portal_x_mm - pad_x)
                stub_y = pad_y + t * (portal_y_mm - pad_y)
                if not self._check_clearance_to_pads(stub_x, stub_y, pad_id, pad_geometries):
                    stub_clear = False
                    break

            if not stub_clear:
                logger.debug(f"Pad {pad_id}: stub path violates clearance")
                continue

            # DRC passed!
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

        # All attempts failed DRC
        logger.warning(f"Pad {pad_id}: failed to find DRC-clean portal after {max_attempts} attempts")
        return None

    def _emit_portal_escape_geometry(self, net_id: str, pad_id: str, portal: Portal, entry_layer: int):
        """Emit vertical escape stub and portal via for a pad"""
        geometry = []

        # 1. Vertical escape stub on pad layer (F.Cu) from pad to portal
        pad_layer_name = self.config.layer_names[portal.pad_layer] if portal.pad_layer < len(self.config.layer_names) else f"L{portal.pad_layer}"

        # Get portal mm coordinates
        portal_x_mm, portal_y_mm = self.lattice.geom.lattice_to_world(portal.x_idx, portal.y_idx)

        # Vertical stub from pad to portal on pad layer
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

    def _check_escape_conflicts(self, portal_geometry: Dict, pad_geometries: Dict) -> List[str]:
        """
        Check for DRC conflicts between escape geometries.

        Returns list of pad_ids that have conflicts and need to be regenerated.
        """
        conflicts = set()

        # Check via-to-via conflicts
        all_vias = []
        for pad_id, (tracks, vias, portal, entry_layer) in portal_geometry.items():
            for via in vias:
                all_vias.append((pad_id, via))

        # Check each via against all other vias
        for i, (pad_id_a, via_a) in enumerate(all_vias):
            for j, (pad_id_b, via_b) in enumerate(all_vias):
                if i >= j:
                    continue

                # Calculate distance between vias
                dx = via_a['x'] - via_b['x']
                dy = via_a['y'] - via_b['y']
                distance = (dx * dx + dy * dy) ** 0.5

                # Via clearance: diameter/2 + diameter/2 + clearance
                via_radius = via_a.get('diameter', 0.25) / 2.0
                required_clearance = 2 * via_radius + PAD_CLEARANCE_MM

                if distance < required_clearance:
                    conflicts.add(pad_id_a)
                    conflicts.add(pad_id_b)
                    logger.debug(f"Via conflict: {pad_id_a} <-> {pad_id_b}, distance={distance:.3f}mm < {required_clearance:.3f}mm")

        # Check track-to-via conflicts
        for pad_id_track, (tracks, _, _, _) in portal_geometry.items():
            for track in tracks:
                tx1, ty1 = track['x1'], track['y1']
                tx2, ty2 = track['x2'], track['y2']

                for pad_id_via, (_, vias, _, _) in portal_geometry.items():
                    if pad_id_track == pad_id_via:
                        continue

                    for via in vias:
                        vx, vy = via['x'], via['y']
                        via_radius = via.get('diameter', 0.25) / 2.0

                        # Calculate distance from via to track
                        dist = self._point_to_segment_distance(vx, vy, tx1, ty1, tx2, ty2)

                        required_clearance = via_radius + PAD_CLEARANCE_MM

                        if dist < required_clearance:
                            conflicts.add(pad_id_track)
                            conflicts.add(pad_id_via)
                            logger.debug(f"Track-via conflict: {pad_id_track} <-> {pad_id_via}, distance={dist:.3f}mm")

        return list(conflicts)

    def _point_to_segment_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2)"""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
