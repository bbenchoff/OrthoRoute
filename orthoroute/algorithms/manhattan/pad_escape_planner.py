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
    axis: str = "y"
    via_x: Optional[float] = None
    via_y: Optional[float] = None
    dynamic_entry: bool = False


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
        self.portal_candidates: Dict[str, List[Portal]] = {}
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
        self.portal_candidates.clear()
        self._occupied_portal_cells = set()

        # STEP 1: Collect all routable pads with grid positions
        # IMPORTANT: Build deterministically sorted list for reproducible results
        pad_list = []  # [(pad_obj, pad_id, x_idx, y_idx, pad_x, pad_y, pad_layer), ...]

        for comp in getattr(board, "components", []):
            for pad in getattr(comp, "pads", []):
                # Skip through-hole pads
                drill = getattr(pad, 'drill', None)
                if drill is None:
                    # Domain Pad objects (file-parser path) name it drill_size
                    drill = getattr(pad, 'drill_size', None) or 0.0
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
            drill = getattr(pad, 'drill', None)
            if drill is None:
                # Domain Pad objects (file-parser path) name it drill_size
                drill = getattr(pad, 'drill_size', None) or 0.0
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

        # Dense columnar SMD connectors have a stronger geometric structure
        # than the generic column heuristic can infer from snapped cells.
        # Preserve their exact pad X coordinate and connect the resulting via
        # to the nearby lattice node on the negotiated entry layer.
        dynamic_pad_ids = self._plan_dynamic_columnar_escapes(
            pad_list,
            pad_geometries,
            spatial_index,
        )

        # STEP 2: Group remaining pads by snapped lattice column.
        columns = {}  # x_idx -> [(pad_obj, pad_id, y_idx, pad_x, pad_y, pad_layer), ...]
        for pad, pad_id, x_idx, y_idx, pad_x, pad_y, pad_layer in pad_list:
            if pad_id in dynamic_pad_ids:
                continue
            if x_idx not in columns:
                columns[x_idx] = []
            columns[x_idx].append((pad, pad_id, y_idx, pad_x, pad_y, pad_layer))

        # STEP 3: Process each column (already sorted from Step 1)
        portal_count = len(dynamic_pad_ids)

        # REVERTED: Back to simple random (no env var complexity)
        # Both portal layers and Y-offsets use random assignment

        for x_idx in sorted(columns.keys()):  # Process columns in deterministic order
            column_pads = columns[x_idx]
            # No need to re-sort: already sorted in Step 1 by (x_idx, y_idx)

            # Plan portals for this column (pass spatial index for fast DRC!)
            portals_created = self._plan_column_escapes(x_idx, column_pads, pad_geometries, spatial_index)
            portal_count += portals_created

        logger.info(f"Planned {portal_count} portals using column-based approach")

        # Preserve the collision-free primary assignment, then collect nearby
        # DRC-valid alternatives for negotiated routing. Alternatives may
        # overlap one another, but never an already assigned primary portal;
        # PathFinder ownership/congestion chooses the final combination.
        self._collect_portal_candidates(
            pad_list, pad_geometries, spatial_index
        )

        # STEP 3.5: Fill gaps in portal distribution (DISABLED)
        # Gap-filling portals didn't solve blank bands and just add visual clutter
        # DISABLED: Set gap_threshold impossibly high so this never triggers
        logger.info(f"[GAP-FILL] Gap-filling DISABLED (portals didn't improve spreading)")
        if False:  # Disabled
            logger.info(f"[GAP-FILL] self.portals exists with {len(self.portals)} items")
            # Get sorted list of X-coordinates with portals
            try:
                portal_x_coords = sorted(set(portal.x_idx for portal in self.portals.values()))
                logger.info(f"[GAP-FILL] Extracted {len(portal_x_coords)} unique X-coordinates")
            except Exception as e:
                logger.error(f"[GAP-FILL] Error extracting X-coordinates: {e}")
                raise

            if len(portal_x_coords) > 1:
                gap_portals_created = 0
                gap_fill_spacing = 1  # DENSE: Create portals every 1 grid step to fully utilize vertical space
                gap_threshold = 2     # AGGRESSIVE: Fill any gap larger than 2 grid steps

                # Entry layer for gap-filling portals (distribute across all routing layers)
                # Use random layer selection to encourage even layer utilization
                # Same contract as pad portals: inner layers only (see above)
                if self.lattice.layers > 2:
                    gap_routing_layers = list(range(1, self.lattice.layers - 1))
                else:
                    gap_routing_layers = [1]
                gap_entry_layer = random.choice(gap_routing_layers)

                logger.info(f"[GAP-FILL] Checking for gaps in portal distribution (threshold={gap_threshold} steps)")

                # Check each consecutive pair of X-coordinates
                for i in range(len(portal_x_coords) - 1):
                    x_left = portal_x_coords[i]
                    x_right = portal_x_coords[i + 1]
                    gap_size = x_right - x_left

                    if gap_size > gap_threshold:
                        logger.info(f"[GAP-FILL] Found gap: X={x_left} to X={x_right} ({gap_size} steps)")

                        # Collect Y-coordinates from portals at left and right boundaries
                        left_y_coords = [p.y_idx for p in self.portals.values() if p.x_idx == x_left]
                        right_y_coords = [p.y_idx for p in self.portals.values() if p.x_idx == x_right]

                        # Use median Y-coordinate as representative
                        if left_y_coords and right_y_coords:
                            left_y_median = sorted(left_y_coords)[len(left_y_coords) // 2]
                            right_y_median = sorted(right_y_coords)[len(right_y_coords) // 2]
                            # Average the two medians for gap portal Y-coordinate
                            gap_y_base = (left_y_median + right_y_median) // 2
                        elif left_y_coords:
                            gap_y_base = sorted(left_y_coords)[len(left_y_coords) // 2]
                        elif right_y_coords:
                            gap_y_base = sorted(right_y_coords)[len(right_y_coords) // 2]
                        else:
                            # Fallback: use middle of board
                            gap_y_base = self.lattice.y_steps // 2

                        # Clamp Y to valid range
                        gap_y_base = max(0, min(gap_y_base, self.lattice.y_steps - 1))

                        # Create portals at evenly-spaced X-coordinates within the gap
                        gap_x_coords = range(x_left + gap_fill_spacing, x_right, gap_fill_spacing)

                        for gap_x in gap_x_coords:
                            # Create synthetic portal ID
                            synthetic_pad_id = f"GAP_PORTAL_X{gap_x}_Y{gap_y_base}"

                            # Convert to world coordinates for portal
                            gap_x_mm, gap_y_mm = self.lattice.geom.lattice_to_world(gap_x, gap_y_base)

                            # Create gap-filling portal (no pad, just a routing anchor point)
                            gap_portal = Portal(
                                x_idx=gap_x,
                                y_idx=gap_y_base,
                                pad_layer=0,  # Nominal pad layer (not used for gap portals)
                                delta_steps=0,  # No escape (this is already a portal point)
                                direction=0,    # No direction (no escape)
                                pad_x=gap_x_mm,
                                pad_y=gap_y_mm,
                                entry_layer=gap_entry_layer,
                                score=0.0,
                                retarget_count=0
                            )

                            # Add to portals dict with synthetic ID
                            # Note: These portals won't generate escape geometry (no real pad)
                            # but they will be available for pathfinding as routing waypoints
                            self.portals[synthetic_pad_id] = gap_portal
                            gap_portals_created += 1

                            logger.debug(f"[GAP-FILL] Created portal at X={gap_x}, Y={gap_y_base} (layer={gap_entry_layer})")

                logger.info(f"[GAP-FILL] Created {gap_portals_created} gap-filling portals")
                portal_count += gap_portals_created

        # Debug: Check portal X-distribution
        if self.portals:
            portal_x_coords = []
            for portal in self.portals.values():
                x = portal.x_idx
                portal_x_coords.append(x)

            unique_x = len(set(portal_x_coords))
            logger.info(f"[PORTAL-DISTRIBUTION] {len(portal_x_coords)} portals across {unique_x} unique X-coordinates")
            logger.info(f"[PORTAL-DISTRIBUTION] X-range: min={min(portal_x_coords)}, max={max(portal_x_coords)}")

            # Show histogram of X-distribution
            import collections
            x_histogram = collections.Counter(portal_x_coords)
            x_bins = sorted(x_histogram.keys())
            logger.info(f"[PORTAL-DISTRIBUTION] First 10 X-coords: {x_bins[:10]}")
            logger.info(f"[PORTAL-DISTRIBUTION] Last 10 X-coords: {x_bins[-10:]}")

            # Check for gaps
            if len(x_bins) > 1:
                max_gap = max(x_bins[i+1] - x_bins[i] for i in range(len(x_bins)-1))
                logger.info(f"[PORTAL-DISTRIBUTION] Largest gap between portal columns: {max_gap} grid steps")

        # Log portal layer distribution
        if hasattr(self, '_layer_counts') and self._layer_counts:
            logger.info(f"[PORTAL-LAYERS] Distribution across {len(self._layer_counts)} layers:")
            for layer in sorted(self._layer_counts.keys()):
                count = self._layer_counts[layer]
                pct = (count / len(self.portals)) * 100 if self.portals else 0
                logger.info(f"[PORTAL-LAYERS]   Layer {layer}: {count} portals ({pct:.1f}%)")

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

    def _plan_dynamic_columnar_escapes(
        self,
        pad_list: List,
        pad_geometries: Dict,
        spatial_index: Dict,
    ) -> set:
        """Plan straight, staggered escapes for regular dense connectors.

        A connector qualifies only when it has many equally populated
        physical X columns of tall SMD pads at the routing pitch. Each column
        is sent away from its centreline and adjacent columns alternate
        between the minimum escape length and one additional grid step.
        The physical via remains at the pad X coordinate; ``x_idx`` is only
        the nearby lattice anchor on the negotiated horizontal entry layer.
        """
        from collections import defaultdict

        by_component = defaultdict(list)
        for entry in pad_list:
            pad = entry[0]
            component_id = getattr(pad, "component_id", None)
            if component_id:
                by_component[component_id].append(entry)

        planned_ids = set()
        pitch = float(self.config.grid_pitch)
        min_steps = int(getattr(
            self.config, "portal_delta_min", 3
        ))

        for component_id, entries in sorted(by_component.items()):
            physical_columns = defaultdict(list)
            for entry in entries:
                physical_columns[round(float(entry[4]), 5)].append(entry)

            ordered_columns = sorted(physical_columns.items())
            if len(ordered_columns) < 8:
                continue

            row_counts = {len(column) for _, column in ordered_columns}
            if len(row_counts) != 1:
                continue
            row_count = next(iter(row_counts))
            if row_count < 2 or row_count % 2:
                continue

            regular_gaps = sum(
                abs((right - left) - pitch) <= pitch * 0.05
                for (left, _), (right, _)
                in zip(ordered_columns, ordered_columns[1:])
            )
            if regular_gaps < 0.75 * (len(ordered_columns) - 1):
                continue

            if any(
                pad_geometries[entry[1]]["height"]
                < 2.0 * pad_geometries[entry[1]]["width"]
                or pad_geometries[entry[1]]["width"] > 0.75 * pitch
                for entry in entries
            ):
                continue

            component_portals = {}
            component_cells = set()
            valid = True
            for column_rank, (_, column) in enumerate(ordered_columns):
                ordered_rows = sorted(column, key=lambda entry: entry[5])
                delta_steps = min_steps + (column_rank % 2)
                for row_rank, entry in enumerate(ordered_rows):
                    (
                        _pad,
                        pad_id,
                        x_idx,
                        y_idx,
                        pad_x,
                        pad_y,
                        pad_layer,
                    ) = entry
                    direction = (
                        -1 if row_rank < row_count // 2 else 1
                    )
                    portal = self._try_create_portal(
                        x_idx,
                        y_idx,
                        direction,
                        delta_steps,
                        pad_id,
                        pad_x,
                        pad_y,
                        pad_layer,
                        1,
                        pad_geometries,
                        spatial_index,
                        claim_cell=False,
                        dynamic_entry=True,
                    )
                    if portal is None:
                        valid = False
                        break
                    cell = (portal.x_idx, portal.y_idx)
                    if (
                        cell in component_cells
                        or cell in self._occupied_portal_cells
                    ):
                        valid = False
                        break
                    component_cells.add(cell)
                    component_portals[pad_id] = portal
                if not valid:
                    break

            if not valid:
                logger.info(
                    "[DYNAMIC-ESCAPE] %s did not admit a complete "
                    "straight staggered assignment; using generic planning",
                    component_id,
                )
                continue

            self.portals.update(component_portals)
            self._occupied_portal_cells.update(component_cells)
            planned_ids.update(component_portals)
            logger.info(
                "[DYNAMIC-ESCAPE] %s: %d pads in %d physical columns",
                component_id,
                len(component_portals),
                len(ordered_columns),
            )

        if planned_ids:
            logger.info(
                "[DYNAMIC-ESCAPE] Planned %d straight off-grid punch-ins",
                len(planned_ids),
            )
        return planned_ids

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
                if dist_above == dist_below:
                    # Equal gaps (uniform pin pitch - the common case): use a
                    # UNIFORM direction so escapes run parallel. Opposing
                    # directions with min_steps=3 meet head-on for pads 6
                    # steps apart (2.54mm headers on the 0.4mm grid), which
                    # placed two portal vias on the SAME grid cell.
                    direction = +1
                else:
                    # Choose direction with more space
                    direction = +1 if dist_above > dist_below else -1
            elif dist_above is not None:
                # Only neighbor above - escape AWAY from it (down)
                direction = -1
            elif dist_below is not None:
                # Only neighbor below - escape AWAY from it (up)
                direction = +1
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
            # LAYER SPREADING FIX: Distribute portals across ALL routing layers (both H and V)
            # This encourages use of all available routing channels, not just vertical layers
            # Previous code restricted to vertical layers only, causing empty horizontal channels
            # With blind/buried vias, PathFinder can route efficiently between any layer pair
            # MUST match build_graph's lateral layers: inner layers only on
            # >2-layer boards (B.Cu gets NO edges there - an entry_layer of
            # layers-1 seeds the portal into a disconnected node!). On
            # 2-layer boards B.Cu (layer 1) is the only routing layer.
            if self.lattice.layers > 2:
                ALL_ROUTING_LAYERS = list(range(1, self.lattice.layers - 1))
            else:
                ALL_ROUTING_LAYERS = [1]
            entry_layer = random.choice(ALL_ROUTING_LAYERS)

            # DEBUG: Log layer assignment for verification
            if not hasattr(self, '_layer_counts'):
                self._layer_counts = {}
            self._layer_counts[entry_layer] = self._layer_counts.get(entry_layer, 0) + 1

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
                           pad_geometries: Dict, spatial_index: Dict = None,
                           claim_cell: bool = True,
                           allow_occupied_cell: Tuple[int, int] = None,
                           axis: str = "y",
                           dynamic_entry: bool = False) -> Optional[Portal]:
        """
        Try to create a portal with given parameters, return None if DRC fails.

        This is the DRC validation function with LOCAL checking for performance.

        DRC Checks:
        1. The complete portal via disk clears every nearby pad.
        2. The complete stub centerline, expanded by half the trace width,
           clears every nearby pad.

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
        if axis == "x":
            x_idx_portal = x_idx + direction * delta_steps
            y_idx_portal = y_idx
        else:
            x_idx_portal = x_idx
            y_idx_portal = y_idx + direction * delta_steps
        if not (
            0 <= x_idx_portal < self.lattice.x_steps
            and 0 <= y_idx_portal < self.lattice.y_steps
        ):
            return None

        # HARD GUARANTEE: one portal via per lattice cell. Two portals on the
        # same cell share a via barrel - permanent overuse no amount of
        # negotiation can resolve (and a DRC violation in the output).
        if not hasattr(self, '_occupied_portal_cells'):
            self._occupied_portal_cells = set()
        portal_cell = (x_idx_portal, y_idx_portal)
        if (
            portal_cell in self._occupied_portal_cells
            and portal_cell != allow_occupied_cell
        ):
            return None

        # Convert portal to world coordinates
        anchor_x_mm, anchor_y_mm = self.lattice.geom.lattice_to_world(
            x_idx_portal, y_idx_portal
        )
        portal_x_mm = (
            pad_x if dynamic_entry and axis == "y" else anchor_x_mm
        )
        portal_y_mm = anchor_y_mm

        clearance = float(
            getattr(self.config, "clearance", PAD_CLEARANCE_MM)
        )
        via_radius = 0.5 * float(
            getattr(self.config, "via_diameter", 0.25)
        )
        track_radius = 0.5 * float(
            getattr(self.config, "track_width", 0.24)
        )

        # Check the complete portal via disk, not only its center.
        if not self._check_clearance_to_pads(
            portal_x_mm,
            portal_y_mm,
            pad_id,
            pad_geometries,
            clearance_mm=clearance + via_radius,
            check_radius=3.0,
            spatial_index=spatial_index,
        ):
            return None

        # Check every point on the exact emitted segments. Sampling allowed
        # short pad crossings to fall between the samples.
        for start, end in self._escape_segments(
            pad_x, pad_y, portal_x_mm, portal_y_mm
        ):
            if not self._check_segment_clearance_to_pads(
                start,
                end,
                pad_id,
                pad_geometries,
                clearance + track_radius,
                spatial_index,
            ):
                return None

        # DRC passed! Claim the cell and create the portal
        if claim_cell:
            self._occupied_portal_cells.add(portal_cell)
        return Portal(
            x_idx=x_idx_portal,
            y_idx=y_idx_portal,
            pad_layer=pad_layer,
            delta_steps=delta_steps,
            direction=direction,
            pad_x=pad_x,
            pad_y=pad_y,
            entry_layer=entry_layer,
            score=0.0,
            retarget_count=0,
            axis=axis,
            via_x=portal_x_mm if dynamic_entry else None,
            via_y=portal_y_mm if dynamic_entry else None,
            dynamic_entry=dynamic_entry,
        )

    def _portal_world(self, portal: Portal) -> Tuple[float, float]:
        """Return the physical via location for a portal."""
        if portal.via_x is not None and portal.via_y is not None:
            return float(portal.via_x), float(portal.via_y)
        return self.lattice.geom.lattice_to_world(
            portal.x_idx, portal.y_idx
        )

    @staticmethod
    def _escape_segments(
        pad_x: float,
        pad_y: float,
        portal_x: float,
        portal_y: float,
    ):
        """Return the shortest orthogonal/45-degree stub centerlines."""
        dx = portal_x - pad_x
        dy = portal_y - pad_y
        if abs(dx) <= 0.01 or abs(dy) <= 0.01:
            return [((pad_x, pad_y), (portal_x, portal_y))]

        if abs(dx) <= abs(dy):
            sign_y = 1 if dy > 0 else -1
            intermediate = (
                pad_x,
                pad_y + dy - sign_y * abs(dx),
            )
        else:
            sign_x = 1 if dx > 0 else -1
            intermediate = (
                pad_x + dx - sign_x * abs(dy),
                pad_y,
            )
        segments = []
        if (
            abs(intermediate[0] - pad_x) > 0.01
            or abs(intermediate[1] - pad_y) > 0.01
        ):
            segments.append(((pad_x, pad_y), intermediate))
        segments.append((intermediate, (portal_x, portal_y)))
        return segments

    @staticmethod
    def _segment_intersects_aabb(start, end, bounds) -> bool:
        """Liang-Barsky intersection, including boundary contact."""
        x0, y0 = start
        x1, y1 = end
        xmin, ymin, xmax, ymax = bounds
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in (
            (-dx, x0 - xmin),
            (dx, xmax - x0),
            (-dy, y0 - ymin),
            (dy, ymax - y0),
        ):
            if abs(p) < 1e-15:
                if q < 0:
                    return False
                continue
            ratio = q / p
            if p < 0:
                if ratio > t1:
                    return False
                t0 = max(t0, ratio)
            else:
                if ratio < t0:
                    return False
                t1 = min(t1, ratio)
        return True

    def _check_segment_clearance_to_pads(
        self,
        start,
        end,
        current_pad_id: str,
        pad_geometries: Dict,
        centerline_clearance: float,
        spatial_index: Dict = None,
    ) -> bool:
        """Check an entire trace centerline against expanded pad rectangles."""
        x0, y0 = start
        x1, y1 = end
        midpoint_x = 0.5 * (x0 + x1)
        midpoint_y = 0.5 * (y0 + y1)
        radius = (
            max(abs(x1 - x0), abs(y1 - y0)) * 0.5
            + centerline_clearance
            + 1.0
        )
        if spatial_index is not None:
            pads_to_check = self._get_nearby_pads(
                midpoint_x,
                midpoint_y,
                radius,
                spatial_index,
                pad_geometries,
            )
        else:
            pads_to_check = pad_geometries.keys()

        for other_pad_id in pads_to_check:
            if other_pad_id == current_pad_id:
                continue
            geom = pad_geometries[other_pad_id]
            half_width = geom["width"] * 0.5 + centerline_clearance
            half_height = geom["height"] * 0.5 + centerline_clearance
            bounds = (
                geom["x"] - half_width,
                geom["y"] - half_height,
                geom["x"] + half_width,
                geom["y"] + half_height,
            )
            if self._segment_intersects_aabb(start, end, bounds):
                return False
        return True

    def _collect_portal_candidates(
        self,
        pad_list: List,
        pad_geometries: Dict,
        spatial_index: Dict,
    ) -> None:
        """Collect a small deterministic set of valid escapes per pad."""
        candidate_limit = max(
            1, int(getattr(self.config, "portal_candidate_count", 6))
        )
        seen_pads = set()

        for (
            _pad,
            pad_id,
            x_idx,
            y_idx,
            pad_x,
            pad_y,
            pad_layer,
        ) in pad_list:
            if pad_id in seen_pads:
                continue
            seen_pads.add(pad_id)
            primary = self.portals.get(pad_id)
            if primary is None:
                continue

            candidates = [primary]
            seen_cells = {(primary.x_idx, primary.y_idx)}
            if primary.dynamic_entry:
                # Vary the straight trace length while preserving the
                # connector's physical escape direction and exact pad X.
                # Entry-layer depth is expanded later by PathFinder.
                for delta in sorted(
                    range(
                        int(getattr(
                            self.config, "portal_delta_min", 3
                        )),
                        int(getattr(
                            self.config, "portal_delta_max", 12
                        )) + 1,
                    ),
                    key=lambda value: (
                        abs(value - primary.delta_steps),
                        value,
                    ),
                ):
                    if len(candidates) >= candidate_limit:
                        break
                    if delta == primary.delta_steps:
                        continue
                    portal = self._try_create_portal(
                        x_idx,
                        y_idx,
                        primary.direction,
                        delta,
                        pad_id,
                        pad_x,
                        pad_y,
                        pad_layer,
                        primary.entry_layer,
                        pad_geometries,
                        spatial_index,
                        claim_cell=False,
                        allow_occupied_cell=(
                            primary.x_idx, primary.y_idx
                        ),
                        dynamic_entry=True,
                    )
                    if portal is None:
                        continue
                    cell = (portal.x_idx, portal.y_idx)
                    if cell in seen_cells:
                        continue
                    seen_cells.add(cell)
                    portal.score = (
                        abs(delta - primary.delta_steps)
                        * float(self.config.grid_pitch)
                    )
                    candidates.append(portal)
                self.portal_candidates[pad_id] = candidates
                continue
            choices = []
            for direction in (primary.direction, -primary.direction):
                for delta in range(3, 13):
                    y_portal = y_idx + direction * delta
                    if not (0 <= y_portal < self.lattice.y_steps):
                        continue
                    rank = (
                        abs(delta - primary.delta_steps),
                        0 if direction == primary.direction else 1,
                        delta,
                    )
                    choices.append((rank, "y", direction, delta))
            for direction in (-1, 1):
                for delta in range(3, 13):
                    x_portal = x_idx + direction * delta
                    if not (0 <= x_portal < self.lattice.x_steps):
                        continue
                    rank = (
                        abs(delta - primary.delta_steps),
                        2,
                        delta,
                    )
                    choices.append((rank, "x", direction, delta))

            for _, axis, direction, delta in sorted(choices):
                if len(candidates) >= candidate_limit:
                    break
                portal = self._try_create_portal(
                    x_idx,
                    y_idx,
                    direction,
                    delta,
                    pad_id,
                    pad_x,
                    pad_y,
                    pad_layer,
                    primary.entry_layer,
                    pad_geometries,
                    spatial_index,
                    # Alternatives are not copper until selected. Global
                    # assignment and negotiated physical history choose a
                    # mutually clear set.
                    claim_cell=False,
                    allow_occupied_cell=(
                        primary.x_idx, primary.y_idx
                    ),
                    axis=axis,
                )
                if portal is None:
                    continue
                cell = (portal.x_idx, portal.y_idx)
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                portal.score = (
                    abs(delta - primary.delta_steps)
                    * float(self.config.grid_pitch)
                    + (
                        float(self.config.grid_pitch)
                        if (
                            axis != primary.axis
                            or direction != primary.direction
                        )
                        else 0.0
                    )
                )
                candidates.append(portal)

            self.portal_candidates[pad_id] = candidates

        counts = [len(v) for v in self.portal_candidates.values()]
        if counts:
            logger.info(
                "[PORTAL-CANDIDATES] %d pads, %.2f candidates/pad "
                "(min=%d, max=%d)",
                len(counts),
                sum(counts) / len(counts),
                min(counts),
                max(counts),
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
                    width, height = self._pad_size_mm(pad.size)
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
                    width, height = self._pad_size_mm(pad.size)
                else:
                    width = 0.5
                    height = 0.5

                geometries[pad_id] = {'x': x, 'y': y, 'width': width, 'height': height}

        return geometries

    @staticmethod
    def _pad_size_mm(size) -> Tuple[float, float]:
        """Return a pad size in mm for both domain and pcbnew objects."""
        if hasattr(size, "x") and hasattr(size, "y"):
            # pcbnew VECTOR2I values use integer nanometres.
            return (
                float(size.x) / 1_000_000.0,
                float(size.y) / 1_000_000.0,
            )
        # File-parser domain Pads already store floating-point millimetres.
        return float(size[0]), float(size[1])

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

    def _emit_portal_escape_geometry(
        self,
        net_id: str,
        pad_id: str,
        portal: Portal,
        entry_layer: int,
        include_via: bool = False,
    ):
        """Emit a pad stub and, after routing, its dynamic grid entry."""
        geometry = []

        # 1. Escape routing: vertical segment + 45-degree segment to portal via
        pad_layer_name = self.config.layer_names[portal.pad_layer] if portal.pad_layer < len(self.config.layer_names) else f"L{portal.pad_layer}"

        # DEBUG: Log to catch layer assignment bug
        if portal.pad_layer != 0:
            logger.error(f"[ESCAPE-LAYER-BUG] net={net_id}, pad={pad_id}: portal.pad_layer={portal.pad_layer} (should be 0!), entry_layer={entry_layer}")
            logger.error(f"[ESCAPE-LAYER-BUG] This will create escape stubs on {pad_layer_name} instead of F.Cu!")

        # Get portal mm coordinates
        portal_x_mm, portal_y_mm = self._portal_world(portal)

        for start, end in self._escape_segments(
            portal.pad_x,
            portal.pad_y,
            portal_x_mm,
            portal_y_mm,
        ):
            geometry.append({
                'net': net_id,
                'layer': pad_layer_name,
                'x1': start[0],
                'y1': start[1],
                'x2': end[0],
                'y2': end[1],
                'width': self.config.track_width,
                'escape': True,
            })

        if include_via and portal.dynamic_entry:
            entry_layer_name = (
                self.config.layer_names[entry_layer]
                if entry_layer < len(self.config.layer_names)
                else f"L{entry_layer}"
            )
            geometry.append({
                "net": net_id,
                "x": portal_x_mm,
                "y": portal_y_mm,
                "from_layer": pad_layer_name,
                "to_layer": entry_layer_name,
                "diameter": self.config.via_diameter,
                "drill": self.config.via_drill,
                "escape": True,
                "dynamic_entry": True,
            })

            anchor_x, anchor_y = self.lattice.geom.lattice_to_world(
                portal.x_idx, portal.y_idx
            )
            if (
                abs(anchor_x - portal_x_mm) > 1e-9
                or abs(anchor_y - portal_y_mm) > 1e-9
            ):
                geometry.append({
                    "net": net_id,
                    "layer": entry_layer_name,
                    "x1": portal_x_mm,
                    "y1": portal_y_mm,
                    "x2": anchor_x,
                    "y2": anchor_y,
                    "width": self.config.track_width,
                    "escape": True,
                })

        return geometry
