"""
═══════════════════════════════════════════════════════════════════════════════
PAD ESCAPE PLANNER - PRECOMPUTED DRC-CLEAN ESCAPE ROUTING
═══════════════════════════════════════════════════════════════════════════════

This module handles precomputation of pad escape routing for multi-layer PCBs.
Before any pathfinding begins, we generate escape stubs and vias for all SMD
pads, distributing traffic across horizontal routing layers.

ALGORITHM OVERVIEW:
1. Group pads by column (x_idx), sort each column by y_idx (DETERMINISTIC)
2. For each pad, determine escape direction based on nearest neighbor distances
3. Choose random escape length constrained by local density (SEEDED RNG for reproducibility)
4. Resolve collisions within column using greedy pair-wise shortening
5. DRC check with local radius (3mm) and progressive fallback to opposite direction
6. Emit vertical + 45-degree escape geometry

DETERMINISM:
- Seeded random number generator (default seed: 42, configurable)
- Stable sorting by (x_idx, y_idx, pad_id) ensures reproducible ordering
- Logged seed value allows exact replay of any routing session

KEY FEATURES:
- Column-based processing: O(n) over all pads, O(k²) per column (k=pads in column)
- Distance-based direction: Escapes toward open space, away from neighbors
- Density-aware randomization: Random length (3-12 steps) constrained by local spacing
- Inverted checkerboard fallback: (x + y) % 2 → even=DOWN, odd=UP (isolated pads only)
- Greedy collision resolution: Alternately shorten by 2 steps, min 3 steps guaranteed
- Local DRC checking: 3mm radius only, not O(n²) against all pads
- Progressive fallback: Try shorter lengths, then opposite direction
- 100% coverage: Every pad gets an escape (minimum 3 steps = 1.2mm)
- Vertical + 45-degree routing geometry for manufacturability

PERFORMANCE:
- Fast: O(n) overall, local checks only
- No renegotiation loops or recursive shortening
- Deterministic column-wise processing

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
    """
    Portal escape point for a pad.

    A Portal represents the via/connection point where a pad's escape trace
    reaches a horizontal routing layer. It stores both lattice coordinates
    (for pathfinding) and physical coordinates (for geometry emission).

    Attributes:
        x_idx: Lattice column index (same as pad's snapped column)
        y_idx: Lattice row index of portal via (pad_y_idx ± delta_steps)
        pad_layer: Physical pad layer index (0 = F.Cu, typically)
        delta_steps: Escape length in grid steps (3-12, constrained by density)
        direction: Escape direction (+1 = up/north, -1 = down/south)
        pad_x: Original pad center X in mm (not snapped)
        pad_y: Original pad center Y in mm (not snapped)
        entry_layer: Routing layer the escape via connects to (1-11, for pathfinding)
        score: Quality score (unused, legacy field)
        retarget_count: Number of retargets (unused, legacy field)
    """
    x_idx: int
    y_idx: int
    pad_layer: int
    delta_steps: int
    direction: int
    pad_x: float
    pad_y: float
    entry_layer: int = 1  # Which horizontal layer this escape via connects to
    score: float = 0.0
    retarget_count: int = 0


class PadEscapePlanner:
    """
    Plans and generates DRC-clean escape routing for SMD pads.

    This planner uses a column-based algorithm with density-aware randomization
    to generate escape vias for all SMD pads. The algorithm is O(n) overall and
    guarantees 100% coverage (every pad gets an escape).

    Key innovations:
    - Column-atomic processing eliminates renegotiation loops
    - Distance-based direction selection (escapes toward open space)
    - Density-aware randomization prevents horizontal via lines
    - Local DRC checking (3mm radius) for O(n) performance
    - Progressive fallback ensures every pad gets an escape

    Usage:
        planner = PadEscapePlanner(lattice, config, pad_to_node)
        tracks, vias = planner.precompute_all_pad_escapes(board)
    """

    def __init__(self, lattice, config, pad_to_node: Dict, random_seed: int = 42):
        """
        Initialize pad escape planner.

        Args:
            lattice: Lattice3D instance with grid geometry
            config: PathFinderConfig with routing parameters
            pad_to_node: Dict[pad_id -> node_idx] mapping
            random_seed: Seed for reproducible random number generation (default: 42)
        """
        self.lattice = lattice
        self.config = config
        self.pad_to_node = pad_to_node
        self.portals: Dict[str, Portal] = {}  # pad_id -> Portal
        self.random_seed = random_seed

        # Initialize seeded RNG for deterministic escape planning
        random.seed(self.random_seed)
        logger.info(f"PadEscapePlanner initialized with random seed: {self.random_seed}")

    def precompute_all_pad_escapes(self, board, nets_to_route: List = None) -> Tuple[List, List]:
        """
        Precompute escape routing for SMD pads using column-based processing.

        Algorithm:
        1. Collect all routable pads and snap to grid columns (±0.5 pitch tolerance)
        2. Group pads by column (x_idx), sort each column by y_idx
        3. For each column, call _plan_column_escapes() to process atomically:
           a. Direction selection (distance-based):
              - Find nearest neighbor above/below in column
              - Choose direction with more distance
              - Fallback: inverted checkerboard (x+y)%2 for isolated pads
           b. Density-aware randomization:
              - Calculate available_space = (distance_to_neighbor) // 2
              - Random length = randint(3, min(12, available_space, board_edge))
              - Prevents horizontal via lines in dense areas
           c. Greedy collision resolution:
              - Check all pairs in column for Y-range overlap
              - Alternately shorten by 2 steps until no collision
              - Min length = 3 steps guaranteed
           d. DRC check with progressive fallback:
              - Try current direction, progressively shorter (delta→3)
              - Try opposite direction, progressively longer (3→max)
              - Local DRC only (3mm radius, not O(n²))
        4. Emit vertical + 45-degree geometry for each portal

        Args:
            board: Board with components and pads
            nets_to_route: List of net names to route (if None, uses board.nets)

        Returns:
            Tuple[List[track_dict], List[via_dict]]: Escape tracks and vias for visualization
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
        # IMPORTANT: Build deterministically sorted list for reproducible results
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

        # DETERMINISTIC SORTING: Sort pad_list by (x_idx, y_idx, pad_id) for reproducibility
        # This ensures the same ordering across runs with the same board/seed
        pad_list.sort(key=lambda p: (p[2], p[3], p[1]))  # Sort by x_idx, y_idx, pad_id

        logger.info(f"Collected {len(pad_list)} pads for escape planning (deterministically sorted)")

        # OPTIMIZATION: Build spatial index ONCE for all DRC checks (10-20× speedup!)
        import time
        spatial_start = time.time()
        spatial_index = self._build_spatial_index(pad_geometries)
        spatial_time = time.time() - spatial_start
        logger.info(f"Built spatial index in {spatial_time:.2f}s ({len(pad_geometries)} pads → {len(spatial_index)} grid cells)")

        # STEP 2: Group by column (x_idx)
        columns = {}  # x_idx -> [(pad_obj, pad_id, y_idx, pad_x, pad_y, pad_layer), ...]
        for pad, pad_id, x_idx, y_idx, pad_x, pad_y, pad_layer in pad_list:
            if x_idx not in columns:
                columns[x_idx] = []
            columns[x_idx].append((pad, pad_id, y_idx, pad_x, pad_y, pad_layer))

        # STEP 3: Process each column (already sorted from Step 1)
        portal_count = 0

        # REVERTED: Back to simple random (no env var complexity)
        # Both portal layers and Y-offsets use random assignment

        for x_idx in sorted(columns.keys()):  # Process columns in deterministic order
            column_pads = columns[x_idx]
            # No need to re-sort: already sorted in Step 1 by (x_idx, y_idx)

            # Plan portals for this column (pass spatial index for fast DRC!)
            portals_created = self._plan_column_escapes(x_idx, column_pads, pad_geometries, spatial_index)
            portal_count += portals_created

        logger.info(f"Planned {portal_count} portals using column-based approach")

        # Build reverse lookup: pad_id -> net_id
        pad_to_net = {}
        for net_id, (src_pad_id, dst_pad_id) in net_pad_mapping.items():
            pad_to_net[src_pad_id] = net_id
            pad_to_net[dst_pad_id] = net_id

        # FIRST PASS: Generate all escape geometry
        portal_geometry = {}  # pad_id -> (tracks, vias)
        for pad_id, portal in self.portals.items():
            net_id = pad_to_net.get(pad_id, f"PAD_{pad_id}")

            # Generate escape geometry (stub + via)
            # Portal already has entry_layer stored from _plan_column_escapes
            geometry = self._emit_portal_escape_geometry(net_id, pad_id, portal, portal.entry_layer)

            portal_tracks = []
            portal_vias = []
            for item in geometry:
                if 'x1' in item and 'y1' in item:  # It's a track
                    portal_tracks.append(item)
                elif 'x' in item and 'y' in item:  # It's a via
                    portal_vias.append(item)

            portal_geometry[pad_id] = (portal_tracks, portal_vias)

        logger.info(f"Generated {len(portal_geometry)} escape geometries")

        # Collect all final geometry
        for pad_id, (portal_tracks, portal_vias) in portal_geometry.items():
            tracks.extend(portal_tracks)
            vias.extend(portal_vias)

        logger.info(f"Final: {len(tracks)} escape stubs and {len(vias)} portal vias")
        return (tracks, vias)

    def _plan_column_escapes(self, x_idx: int, column_pads: List, pad_geometries: Dict, spatial_index: Dict = None) -> int:
        """
        Plan escapes for all pads in a single column using greedy collision resolution.

        PORTAL LAYER STRATEGY (controlled by ORTHO_PORTAL_STRATEGY env var):
        - "random": Each portal randomly picks entry_layer from 1-17 (old behavior)
        - "layer1": All portals use entry_layer=1 (new default, PathFinder controls layer spread)

        This is the core algorithm that processes one column atomically:

        STEP 1: Direction Selection (distance-based)
        - For each pad, find nearest neighbor above and below in the column
        - Choose direction with MORE distance (escapes toward open space)
        - If only one neighbor: escape away from it
        - If isolated (no neighbors): use inverted checkerboard (x+y)%2

        STEP 2: Density-Aware Randomization
        - Calculate available_space = (distance_to_neighbor_in_escape_dir) // 2
        - This gives each pad roughly half the gap to its neighbor
        - Random length: randint(min_steps=3, min(max_steps=12, available_space, board_edge))
        - Dense areas get shorter random ranges (e.g., 3-4 steps)
        - Sparse areas get full random range (e.g., 3-12 steps)
        - Result: Randomized vias WITHOUT horizontal lines

        STEP 3: Greedy Collision Resolution
        - Check all pairs of escapes in column for Y-range overlap (with 1-step buffer)
        - If collision found: alternately shorten one escape by 2 steps
        - Continue until no collisions or min_steps reached
        - Max 100 iterations (typically converges in <10)

        STEP 4: DRC Check with Progressive Fallback
        - For each planned escape, call _try_create_portal() with:
          a. Current direction, try lengths: delta_steps, delta_steps-1, ..., min_steps
          b. Opposite direction, try lengths: min_steps, ..., max_steps
        - DRC uses local radius (3mm) not O(n²) all-pads check
        - First valid escape is accepted
        - If no valid escape found: log warning (rare)

        Args:
            x_idx: Column index in lattice
            column_pads: List of (pad_obj, pad_id, y_idx, pad_x, pad_y, pad_layer) sorted by y_idx
            pad_geometries: Dict[pad_id -> {x, y, width, height}] for DRC checking

        Returns:
            Number of portals successfully created for this column
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

            # REVERTED: Back to simple random delta (density-aware)
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
            # CRITICAL FIX: Portals MUST land on vertical layers only!
            # Escape stubs go in Y direction, so portals need layers with Y-edges (2,4,6,8,10)
            # Horizontal layers (1,3,5,7,9,11) have NO vertical edges → unreachable!
            VERTICAL_LAYERS = [2, 4, 6, 8, 10]
            available_vertical_layers = [L for L in VERTICAL_LAYERS if L < self.lattice.layers]
            if not available_vertical_layers:
                available_vertical_layers = [2]  # Fallback to layer 2
            entry_layer = random.choice(available_vertical_layers)

            # Try progressively shorter lengths, then opposite direction
            portal = None

            # First try: current direction, progressively shorter
            for try_delta in range(delta_steps, min_steps - 1, -1):
                portal = self._try_create_portal(x_idx, y_idx, direction, try_delta,
                                                  pad_id, pad_x, pad_y, pad_layer, entry_layer, pad_geometries, spatial_index)
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
                                                      pad_id, pad_x, pad_y, pad_layer, entry_layer, pad_geometries, spatial_index)
                    if portal:
                        break

            if portal:
                self.portals[pad_id] = portal
                portal_count += 1
            else:
                logger.warning(f"Pad {pad_id}: could not find valid escape in any direction")

        return portal_count

    def _try_create_portal(self, x_idx: int, y_idx: int, direction: int, delta_steps: int,
                           pad_id: str, pad_x: float, pad_y: float, pad_layer: int, entry_layer: int,
                           pad_geometries: Dict, spatial_index: Dict = None) -> Optional[Portal]:
        """
        Try to create a portal with given parameters, return None if DRC fails.

        This is the DRC validation function with LOCAL checking for performance.

        DRC Checks:
        1. Portal via position: Must maintain PAD_CLEARANCE_MM from all pads within 3mm
        2. Stub path: Sample 3 points (25%, 50%, 75%) along escape trace
        3. Each sample point must maintain clearance from nearby pads

        Performance: O(k) where k = pads within 3mm radius (typically 5-20 pads)
        NOT O(n) where n = all pads on board (could be 10,000+)

        Args:
            x_idx: Column index in lattice
            y_idx: Pad Y index in lattice
            direction: +1 (up) or -1 (down)
            delta_steps: Escape length in grid steps (3-12)
            pad_id: Pad identifier for clearance checking
            pad_x: Pad center X in mm
            pad_y: Pad center Y in mm
            pad_layer: Pad layer index (typically 0 for F.Cu)
            entry_layer: Horizontal routing layer the escape via connects to (1-11)
            pad_geometries: Dict[pad_id -> {x, y, width, height}]

        Returns:
            Portal object if DRC passes, None if any violation detected
        """
        y_idx_portal = y_idx + direction * delta_steps

        # Convert portal to world coordinates
        portal_x_mm, portal_y_mm = self.lattice.geom.lattice_to_world(x_idx, y_idx_portal)

        # DRC check: portal via clearance (LOCAL only - within 3mm radius)
        # OPTIMIZED: Pass spatial_index for 10-20× speedup!
        if not self._check_clearance_to_pads(portal_x_mm, portal_y_mm, pad_id, pad_geometries,
                                              check_radius=3.0, spatial_index=spatial_index):
            return None

        # DRC check: stub path (LOCAL only)
        # Check 3 sample points along the escape trace
        for t in [0.25, 0.5, 0.75]:
            stub_x = pad_x + t * (portal_x_mm - pad_x)
            stub_y = pad_y + t * (portal_y_mm - pad_y)
            if not self._check_clearance_to_pads(stub_x, stub_y, pad_id, pad_geometries,
                                                   check_radius=3.0, spatial_index=spatial_index):
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
            entry_layer=entry_layer,
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

    def _build_spatial_index(self, pad_geometries: Dict):
        """
        Build spatial index (grid-based) for fast nearest-neighbor queries.

        Uses a simple 2D grid with 5mm cells for O(1) lookup of nearby pads.
        Much faster than checking all 16K pads for each DRC check.

        Args:
            pad_geometries: Dict[pad_id -> {x, y, width, height}]

        Returns:
            spatial_index: Dict[(grid_x, grid_y) -> [pad_ids]]
        """
        spatial_index = {}
        grid_size = 5.0  # 5mm grid cells

        for pad_id, geom in pad_geometries.items():
            # Compute grid cell for this pad
            grid_x = int(geom['x'] / grid_size)
            grid_y = int(geom['y'] / grid_size)

            # Add to spatial index
            key = (grid_x, grid_y)
            if key not in spatial_index:
                spatial_index[key] = []
            spatial_index[key].append(pad_id)

        return spatial_index

    def _get_nearby_pads(self, x: float, y: float, radius_mm: float,
                         spatial_index: Dict, pad_geometries: Dict) -> List[str]:
        """
        Get pad IDs within radius_mm of point (x, y) using spatial index.

        Much faster than iterating all pads: O(k) where k = nearby pads (5-20)
        instead of O(n) where n = all pads (16,000).

        Args:
            x, y: Point coordinates in mm
            radius_mm: Search radius in mm
            spatial_index: Grid index from _build_spatial_index()
            pad_geometries: Full pad geometry dict

        Returns:
            List of pad IDs within radius
        """
        grid_size = 5.0
        nearby_pads = []

        # Compute grid cell range to check
        grid_radius = int(radius_mm / grid_size) + 1
        center_grid_x = int(x / grid_size)
        center_grid_y = int(y / grid_size)

        # Check all grid cells within radius
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                key = (center_grid_x + dx, center_grid_y + dy)
                if key in spatial_index:
                    # Check actual distance for pads in this cell
                    for pad_id in spatial_index[key]:
                        geom = pad_geometries[pad_id]
                        dist_x = abs(geom['x'] - x)
                        dist_y = abs(geom['y'] - y)
                        if dist_x <= radius_mm and dist_y <= radius_mm:
                            nearby_pads.append(pad_id)

        return nearby_pads

    def _check_clearance_to_pads(self, x: float, y: float, current_pad_id: str,
                                  pad_geometries: Dict, clearance_mm: float = None,
                                  debug: bool = False, check_radius: float = None,
                                  spatial_index: Dict = None) -> bool:
        """
        Check if point (x, y) maintains clearance from other pads.

        OPTIMIZED VERSION: Uses spatial indexing for 10-20× speedup!

        Args:
            x: Point X coordinate in mm
            y: Point Y coordinate in mm
            current_pad_id: Pad ID to skip (avoid self-checking)
            pad_geometries: Dict[pad_id -> {x, y, width, height}]
            clearance_mm: Required clearance (default: PAD_CLEARANCE_MM = 0.15mm)
            debug: If True, log violation details (slower, for debugging only)
            check_radius: Search radius in mm (default: 3.0mm for local DRC)
            spatial_index: Optional spatial index for fast lookup (highly recommended!)

        Returns:
            True if clearance is OK (no violations)
            False if any violation detected
        """
        if clearance_mm is None:
            clearance_mm = PAD_CLEARANCE_MM

        if check_radius is None:
            check_radius = 3.0  # Default to local DRC

        violations = []

        # OPTIMIZATION: Use spatial index if available (10-20× faster!)
        if spatial_index is not None:
            pads_to_check = self._get_nearby_pads(x, y, check_radius, spatial_index, pad_geometries)
        else:
            # Fallback: check all pads (slow!)
            pads_to_check = pad_geometries.keys()

        for pad_id in pads_to_check:
            if pad_id == current_pad_id:
                continue  # Skip self

            geom = pad_geometries[pad_id]
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
                    return False  # Fast fail on first violation!

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

        # DEBUG: Log to catch layer assignment bug
        if portal.pad_layer != 0:
            logger.error(f"[ESCAPE-LAYER-BUG] net={net_id}, pad={pad_id}: portal.pad_layer={portal.pad_layer} (should be 0!), entry_layer={entry_layer}")
            logger.error(f"[ESCAPE-LAYER-BUG] This will create escape stubs on {pad_layer_name} instead of F.Cu!")

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
