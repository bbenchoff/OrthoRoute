"""
═══════════════════════════════════════════════════════════════════════════════
PAD ESCAPE PLANNER - PRECOMPUTED DRC-CLEAN ESCAPE ROUTING
═══════════════════════════════════════════════════════════════════════════════

This module handles precomputation of pad escape routing for multi-layer PCBs.
Before any pathfinding begins, we generate escape stubs and vias for all SMD
pads, distributing traffic across horizontal routing layers.

KEY FEATURES:
- Checkerboard direction pattern: (x + y) % 2 → even=UP, odd=DOWN (strict enforcement)
- Random escape lengths (1.2mm - 4.8mm / 3-12 grid steps)
- Column overlap prevention (tracks existing Y-ranges per column)
- Smart conflict resolution: iterations 1-10 shorten overlapping blockers (2→4→6→8...), after 10 remove blocker
- Only considers blockers that overlap with target escape range (correct by construction)
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
        self.column_ranges: Dict[int, List[Tuple[int, int, str]]] = {}  # x_idx -> [(y_min, y_max, pad_id), ...]

    def precompute_all_pad_escapes(self, board, nets_to_route: List = None) -> Tuple[List, List]:
        """
        Precompute escape routing for SMD pads attached to nets we want to route.

        For each SMD pad on a routable net:
        1. Snap X to nearest grid column (±½ pitch allowed)
        2. Pick direction via checkerboard: (x_idx + y_idx) % 2 → even=UP, odd=DOWN
        3. Pick random vertical length d ∈ {3..12} grid steps (1.2mm - 4.8mm @ 0.4mm pitch)
        4. Check overlap with existing escapes on same column
        5. If blocked, progressively shorten longest blocker (2→4→6→8 steps, min 3)
        6. DRC check: ensure stub and via maintain clearance from other pads
        7. Compute stub tip (xg, yg±d) on F.Cu
        8. Place via to random horizontal layer (odd index: In1, In3, ..., B.Cu)

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
        self.column_ranges.clear()

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

        logger.info(f"Planned {portal_count} portals for routable nets (using checkerboard pattern)")

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

                # Remove old portal's range from column_ranges before regenerating
                if pad_id in self.portals:
                    old_portal = self.portals[pad_id]
                    old_x_idx = old_portal.x_idx
                    if old_x_idx in self.column_ranges:
                        self.column_ranges[old_x_idx] = [
                            (y_min, y_max, pid) for (y_min, y_max, pid) in self.column_ranges[old_x_idx]
                            if pid != pad_id
                        ]

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
        Plan portal escape with CHECKERBOARD direction and random length.

        Direction: (x_idx + y_idx) % 2 → even=UP, odd=DOWN (strict)
        Length: Random 3-12 grid steps, avoiding overlaps on same column
        Blocked: Shorten longest blocker by 4 steps (min 3), retry
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

        # STEP 1: Checkerboard direction - (x + y) % 2 determines UP or DOWN (STRICT)
        checkerboard_value = (x_idx + y_idx_pad) % 2
        direction = +1 if checkerboard_value == 0 else -1  # Even=UP, Odd=DOWN

        # STEP 2: Find valid length range in checkerboard direction
        min_steps = 3
        max_steps = 12

        # Calculate max possible steps in checkerboard direction
        if direction > 0:
            max_possible = self.lattice.y_steps - 1 - y_idx_pad
        else:
            max_possible = y_idx_pad

        # Clamp max_steps to what's physically possible
        max_steps_bounded = min(max_steps, max_possible)

        if max_steps_bounded < min_steps:
            logger.debug(f"Pad {pad_id}: insufficient space in checkerboard direction")
            return None

        # STEP 3: Try multiple random lengths in the checkerboard direction
        max_attempts = 20
        blocking_ranges = []

        for attempt in range(max_attempts):
            # Random length within valid range
            delta_steps = random.randint(min_steps, max_steps_bounded)

            # Calculate portal position
            y_idx_portal = y_idx_pad + direction * delta_steps

            # Check for overlap with existing escapes on same column
            y_min = min(y_idx_pad, y_idx_portal)
            y_max = max(y_idx_pad, y_idx_portal)

            if x_idx in self.column_ranges:
                overlaps = False
                for (existing_y_min, existing_y_max, existing_pad_id) in self.column_ranges[x_idx]:
                    # Check if ranges overlap (with 1-step buffer)
                    if not (y_max + 1 < existing_y_min or y_min - 1 > existing_y_max):
                        overlaps = True
                        blocking_ranges.append((existing_y_min, existing_y_max, existing_pad_id))
                        break

                if overlaps:
                    continue  # Try different random length

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

            # DRC passed! Record the Y-range for this column
            if x_idx not in self.column_ranges:
                self.column_ranges[x_idx] = []
            self.column_ranges[x_idx].append((y_min, y_max, pad_id))

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

        # All attempts failed - try to shorten blocking escapes
        if blocking_ranges:
            logger.info(f"Pad {pad_id}: blocked by {len(set(blocking_ranges))} escapes, attempting to shorten one")
            return self._shorten_blocker_and_retry(pad, pad_id, pad_geometries, x_idx, y_idx_pad, direction)

        # All attempts failed DRC
        logger.warning(f"Pad {pad_id}: failed to find DRC-clean portal after {max_attempts} attempts")
        return None

    def _shorten_blocker_and_retry(self, pad, pad_id: str, pad_geometries: Dict,
                                     x_idx: int, y_idx_pad: int, direction: int) -> Optional[Portal]:
        """
        Shorten a blocking escape and retry current pad.
        Strategy: Try shortening for first 10 iterations, then switch to removal.
        Only considers blockers that overlap with our target escape range.
        """
        # Find blocking pads from column_ranges
        if x_idx not in self.column_ranges or not self.column_ranges[x_idx]:
            return None

        # Find the NEAREST blocking pad in our direction
        # We need to find the blocker that's actually preventing us from escaping
        min_steps = 3
        max_steps = 12

        # Calculate the range we're trying to escape into
        if direction > 0:
            target_y_min = y_idx_pad + min_steps
            target_y_max = y_idx_pad + min(max_steps, self.lattice.y_steps - 1 - y_idx_pad)
        else:
            target_y_min = max(0, y_idx_pad - max_steps)
            target_y_max = y_idx_pad - min_steps

        # Find blockers that OVERLAP with our target escape range
        blocking_candidates = []
        for (existing_y_min, existing_y_max, existing_pad_id) in self.column_ranges[x_idx]:
            # Skip self
            if existing_pad_id == pad_id:
                continue

            # Check if this range overlaps with our target range (with buffer)
            if not (target_y_max + 1 < existing_y_min or target_y_min - 1 > existing_y_max):
                blocking_candidates.append(existing_pad_id)

        if not blocking_candidates:
            return None

        # Check current recursion depth
        if not hasattr(self, '_shorten_depth'):
            self._shorten_depth = 0

        # OPTION 3: After 10 iterations, switch from shortening to complete removal
        if self._shorten_depth >= 10:
            # Column is too dense - remove one blocker completely
            victim_pad_id = blocking_candidates[0]
            logger.info(f"Pad {pad_id}: depth={self._shorten_depth}, column too dense, removing {victim_pad_id} completely")

            # Remove victim's portal
            if victim_pad_id in self.portals:
                del self.portals[victim_pad_id]

                # Add to replan queue
                if not hasattr(self, 'pads_needing_replan'):
                    self.pads_needing_replan = set()
                self.pads_needing_replan.add(victim_pad_id)

                # Remove from column_ranges
                if x_idx in self.column_ranges:
                    self.column_ranges[x_idx] = [
                        (y_min, y_max, pid) for (y_min, y_max, pid) in self.column_ranges[x_idx]
                        if pid != victim_pad_id
                    ]

                # Retry current pad recursively
                if self._shorten_depth < 15:  # Higher limit for removal mode
                    self._shorten_depth += 1
                    result = self._plan_random_portal_for_pad_with_drc(pad, pad_id, pad_geometries, debug=False)
                    self._shorten_depth -= 1
                    return result

            logger.debug(f"Pad {pad_id}: could not remove victim")
            return None

        # SHORTENING MODE (depth < 10): Pick the blocker with the LONGEST escape
        best_blocker = None
        best_length = 0
        for candidate_id in blocking_candidates:
            if candidate_id in self.portals:
                candidate_portal = self.portals[candidate_id]
                if candidate_portal.delta_steps > best_length and candidate_portal.delta_steps > 3:
                    best_blocker = candidate_id
                    best_length = candidate_portal.delta_steps

        # If no blocker can be shortened, completely remove one
        if not best_blocker:
            # All blockers are at minimum - pick any one to completely remove
            if blocking_candidates:
                victim_pad_id = blocking_candidates[0]
                logger.info(f"Pad {pad_id}: all blockers at minimum, removing {victim_pad_id} completely for replanning")

                # Remove victim's portal
                if victim_pad_id in self.portals:
                    del self.portals[victim_pad_id]

                    # Add to replan queue
                    if not hasattr(self, 'pads_needing_replan'):
                        self.pads_needing_replan = set()
                    self.pads_needing_replan.add(victim_pad_id)

                    # Remove from column_ranges
                    if x_idx in self.column_ranges:
                        self.column_ranges[x_idx] = [
                            (y_min, y_max, pid) for (y_min, y_max, pid) in self.column_ranges[x_idx]
                            if pid != victim_pad_id
                        ]

                    # Retry current pad recursively
                    if self._shorten_depth < 20:
                        self._shorten_depth += 1
                        result = self._plan_random_portal_for_pad_with_drc(pad, pad_id, pad_geometries, debug=False)
                        self._shorten_depth -= 1
                        return result

            logger.debug(f"Pad {pad_id}: no blockers to remove")
            return None

        # Get the blocking portal
        blocker_portal = self.portals[best_blocker]

        # Try progressively larger shortening amounts: 2, 4, 6, ...
        # Track how many times we've shortened (to know which amount to try)
        if not hasattr(self, '_shorten_attempts'):
            self._shorten_attempts = {}

        blocker_key = (x_idx, best_blocker)
        shorten_attempt = self._shorten_attempts.get(blocker_key, 0)
        shorten_amount = 2 + (shorten_attempt * 2)  # 2, 4, 6, 8, ...

        self._shorten_attempts[blocker_key] = shorten_attempt + 1

        # Shorten it by moving it back
        new_delta_steps = max(3, blocker_portal.delta_steps - shorten_amount)

        if new_delta_steps >= blocker_portal.delta_steps:
            # Can't shorten anymore
            logger.debug(f"Pad {pad_id}: can't shorten blocker {best_blocker} (already at {blocker_portal.delta_steps})")
            return None

        logger.info(f"Pad {pad_id}: shortening blocker {best_blocker} from {blocker_portal.delta_steps} to {new_delta_steps} steps (attempt {shorten_attempt + 1}, amount={shorten_amount})")

        # Recalculate blocker's portal position with new shorter length
        _, blocker_y_idx_pad = self.lattice.world_to_lattice(blocker_portal.pad_x, blocker_portal.pad_y)
        new_y_idx_portal = blocker_y_idx_pad + blocker_portal.direction * new_delta_steps

        # Update the blocker's portal
        blocker_portal.delta_steps = new_delta_steps
        blocker_portal.y_idx = new_y_idx_portal

        # Rebuild column_ranges for this column (remove old blocker range)
        self.column_ranges[x_idx] = [
            (y_min, y_max, pid) for (y_min, y_max, pid) in self.column_ranges[x_idx]
            if pid != best_blocker
        ]

        # Re-add shortened blocker's range
        new_y_min = min(blocker_y_idx_pad, new_y_idx_portal)
        new_y_max = max(blocker_y_idx_pad, new_y_idx_portal)
        self.column_ranges[x_idx].append((new_y_min, new_y_max, best_blocker))

        # Retry current pad recursively
        if self._shorten_depth < 20:  # Higher limit to allow removal mode
            self._shorten_depth += 1
            result = self._plan_random_portal_for_pad_with_drc(pad, pad_id, pad_geometries, debug=False)
            self._shorten_depth -= 1

            # If successful, clean up the shorten_attempts for this blocker
            if result:
                if blocker_key in self._shorten_attempts:
                    del self._shorten_attempts[blocker_key]

            return result

        logger.warning(f"Pad {pad_id}: hit recursion limit of 20, giving up")
        return None

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
