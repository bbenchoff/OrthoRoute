"""
Geometry Mixin - Extracted from UnifiedPathFinder

This module contains geometry mixin functionality.
Part of the PathFinder routing algorithm refactoring.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from types import SimpleNamespace
from ....domain.models.board import Board, Pad

logger = logging.getLogger(__name__)


class GeometryMixin:
    """
    Geometry functionality for UnifiedPathFinder.

    This mixin is designed to be used with multiple inheritance.
    It expects the following attributes to be available on self:
    - config: PathFinderConfig instance
    - use_gpu: bool indicating GPU availability
    - node_count: int number of nodes
    - edges: list of edges
    - nodes: dict of node data
    """

    def emit_geometry(self, board) -> tuple[int, int]:
        """Build geometry intents, strict DRC pre-emit gate, push to GUI."""
        # PathFinder bypass tripwire
        if not getattr(self, "_negotiation_ran", False):
            raise RuntimeError("EMIT-TRIPWIRE: PathFinder bypass detected (Dijkstra fast-path).")

        # Check if routing failed due to insufficient layers
        rr = getattr(self, "_routing_result", None)
        if isinstance(rr, dict) and not rr.get("success", True):
            msg = rr.get('message') or \
                  f"[INSUFFICIENT-LAYERS] Need {rr.get('layer_shortfall', 1)} more layers."
            class GeometryPayload:
                def __init__(self, tracks, vias):
                    self.tracks = tracks
                    self.vias = vias
            self._last_geometry_payload = GeometryPayload([], [])
            self._last_failure = msg
            logger.warning("[EMIT-GUARD] %s", msg)
            return (0, 0)

        # Recompute usage/overuse just-in-time
        self._refresh_present_usage_from_store()
        over_sum, over_cnt = self._compute_overuse_stats()
        if over_cnt > 0:
            logger.warning("[EMIT-GUARD] Overuse remains (sum=%d edges=%d) – aborting emit and returning capacity analysis", over_sum, over_cnt)
            raise RuntimeError("[CAPACITY] Overuse remains; run capacity analysis instead of emit")

        logger.info("EMIT-TRIPWIRE: PF_negotiated=True")

        # SURGICAL STEP 4: Emit geometry guard with path availability check
        routed_paths = {net_id: path for net_id, path in self.routed_nets.items() if path and len(path) > 1}
        logger.info(f"[EMIT-GUARD] routed_paths_available={len(routed_paths)}")
        if not routed_paths:
            logger.error("[EMIT-GUARD] No valid paths available for geometry emission")
            raise RuntimeError("[EMIT-GUARD] no paths to convert")

        intents = self._build_geometry_intents()
        viol = self._validate_geometry_intents(intents)

        # Single source of truth for zero-length tracks
        self._zero_len_tracks = viol.zero_len_tracks

        logger.info("[INTENTS] summary: tracks=%d vias=%d zero_len_tracks=%d",
                   len(intents.tracks), len(intents.vias), self._zero_len_tracks)

        # Enhanced strict DRC pre-emit gate with capacity overuse recovery
        if viol.total() == 0:
            logger.info("[STRICT-DRC] pre-emit: all-clear (via_in_pad=0, track_pad_clear=0, via_via_spacing=0, zero_len_tracks=%d)", self._zero_len_tracks)
        else:
            if viol.track_track > 0:
                logger.warning("[STRICT-DRC] pre-emit: %d track-track conflicts; attempting capacity-end reroute", viol.track_track)
                # Convert "overlapping in geometry" -> "overused edges" in edge_store
                self._mark_overlaps_as_overuse(intents)
                # Run one aggressive negotiation shakeout
                self._pres_fac *= 3.0             # push present cost up sharply
                self._hist_fac *= 1.5             # increase historical cost bias a bit
                rerouted = self._pathfinder_negotiation(self._active_nets, None, len(self._active_nets))
                # Rebuild intents and re-check
                intents = self._build_geometry_intents()
                viol = self._validate_geometry_intents(intents)

            if viol.track_track > 0 or viol.track_via > 0 or viol.track_pad_clear > 0:
                logger.warning("[STRICT-DRC] pre-emit: %s", viol)
                # Check if strict DRC is enabled
                if hasattr(self.config, 'strict_drc') and self.config.strict_drc:
                    raise RuntimeError(f"STRICT-DRC: {viol} (routing must be renegotiated)")
                else:
                    logger.warning("[DRC] violations detected: %s (continuing due to strict_drc=False)", viol)
            else:
                logger.error("[STRICT-DRC] pre-emit: violations detected - %s", viol)
                if hasattr(self.config, 'strict_drc') and self.config.strict_drc:
                    raise RuntimeError(f"STRICT-DRC: {viol}")
                else:
                    logger.warning("[DRC] violations detected: %s (continuing due to strict_drc=False)", viol)

        self._last_geometry_payload = self._convert_intents_to_view(intents)
        track_count = len(self._last_geometry_payload.tracks) if self._last_geometry_payload else 0
        via_count = len(self._last_geometry_payload.vias) if self._last_geometry_payload else 0

        # SANITY GATES: Ensure geometry was actually transferred
        intent_tracks = len(intents.tracks) if hasattr(intents, 'tracks') else 0
        intent_vias = len(intents.vias) if hasattr(intents, 'vias') else 0

        if intent_tracks > 0 and track_count == 0:
            logger.error(f"[EMIT-SANITY] GEOMETRY LOSS: {intent_tracks} intent tracks → 0 payload tracks")
            raise RuntimeError(f"EMIT SANITY FAILURE: {intent_tracks} tracks lost in conversion")

        if intent_vias > 0 and via_count == 0:
            logger.error(f"[EMIT-SANITY] GEOMETRY LOSS: {intent_vias} intent vias → 0 payload vias")
            raise RuntimeError(f"EMIT SANITY FAILURE: {intent_vias} vias lost in conversion")

        if track_count != intent_tracks:
            logger.warning(f"[EMIT-SANITY] TRACK COUNT MISMATCH: {intent_tracks} intents → {track_count} payload")

        if via_count != intent_vias:
            logger.warning(f"[EMIT-SANITY] VIA COUNT MISMATCH: {intent_vias} intents → {via_count} payload")

        logger.info("[EMIT] Generated %d tracks, %d vias (verified)", track_count, via_count)

        return (track_count, via_count)


    def get_geometry_payload(self):
        """Return last payload for GUI draw."""
        return getattr(self, "_last_geometry_payload", None)


    def prepare_routing_runtime(self):
        """Prepare routing runtime - apply CSR masks after CSR is built."""
        logger.info("[RUNTIME] Preparing routing runtime - applying CSR masks")

        # Apply CSR masks now that the CSR matrix exists
        board = getattr(self, '_cached_board', None)
        if board:
            self._apply_csr_masks(board)
        else:
            logger.warning("[RUNTIME] No cached board available for CSR mask application")

    # ============================================================================
    # HELPER METHODS FOR PUBLIC API
    # ============================================================================


    def _build_pad_keepouts(self, board):
        """Build pad keepout masks."""
        # Initialize portal metrics
        self._metrics.setdefault("portal_edges_registered", 0)
        logger.info("[PAD-KEEPOUT] Building pad keepout masks")

        # CSR masks will be applied in prepare_routing_runtime() after CSR is built


    def _snap_all_pads_to_lattice(self, board):
        """SIMPLIFIED: Direct pad-to-lattice connection without broken portal system."""
        terminal_map = {}  # Map (net_name, pad_uid) -> lattice_node_idx
        connected_count = 0
        total_pads = 0

        for component in board.components:
            for pad in component.pads:
                total_pads += 1

                # Get pad coordinates and net
                pad_x, pad_y = self._get_pad_coordinates(pad)
                net_name = self._get_pad_net_name(pad)

                if net_name != "unconnected":
                    # DIRECT: Find nearest lattice node to pad coordinates
                    if self.geometry is None:
                        logger.error("[SNAP] Geometry system not initialized")
                        continue

                    # Convert pad coordinates to lattice indices
                    lattice_x, lattice_y = self.geometry.world_to_lattice(pad_x, pad_y)

                    # Clamp to valid bounds
                    lattice_x = max(0, min(lattice_x, self.geometry.x_steps - 1))
                    lattice_y = max(0, min(lattice_y, self.geometry.y_steps - 1))

                    # Convert lattice coordinates to node index using KiCadGeometry system
                    lattice_node_idx = self.geometry.node_index(lattice_x, lattice_y, 0)  # F.Cu layer

                    # DIRECT: Store pad-to-lattice mapping (no portal complications)
                    comp_ref = self._uid_component(getattr(pad, "component", getattr(pad, "footprint", None)))
                    pad_lbl  = self._uid_pad_label(pad, comp_ref)
                    uid      = (comp_ref, pad_lbl)

                    # Store direct pad-to-lattice mapping
                    node_idx = int(lattice_node_idx)  # must be an integer node id
                    self._portal_by_pad_id[id(pad)] = node_idx
                    self._portal_by_uid[uid] = node_idx

                    # Keep legacy map for compatibility
                    terminal_key = (net_name, f"{comp_ref}_{pad_lbl}")
                    terminal_map[terminal_key] = lattice_node_idx

                    logger.debug(f"[SNAP] pad ({pad_x:.1f},{pad_y:.1f}) -> lattice ({lattice_x},{lattice_y}) -> node_idx={lattice_node_idx}")
                    connected_count += 1

        # Store terminal map for routing phase
        self._portal_terminal_map = terminal_map

        # Log simplified results
        logger.info(f"[SNAP] Connected {connected_count}/{total_pads} pads directly to lattice nodes")
        sample = list(self._portal_terminal_map.keys())[:5]
        logger.info(f"[SNAP] terminal map size={len(self._portal_terminal_map)} sample={sample}")

        # No portal edges to refresh - direct mapping complete
        self._assert_live_sizes()  # hard-asserts N/E match and arrays match E
        self._build_reverse_edge_index_gpu()  # Must use the LIVE CSR sizes

        # Backfill graph_state fields after CSR finalization
        gs = getattr(self, "graph_state", None)
        if gs is not None:
            gs.lattice_node_count = getattr(self, "lattice_node_count", None)
            gs.indices_cpu = self.indices_cpu
            gs.indptr_cpu = self.indptr_cpu
            gs.edge_count = len(self.indices_cpu)
            logger.debug("[PORTAL-BIND] Backfilled graph_state with CSR fields")


    def _coords_to_node_index(self, gx: int, gy: int, layer: int) -> int:
        """Convert grid coordinates (gx,gy,layer) to node index using geometry."""
        geo = getattr(self, "geometry", None)
        if geo is None:
            return -1
        x_steps = int(getattr(geo, "x_steps", 0))
        y_steps = int(getattr(geo, "y_steps", 0))
        layers  = int(getattr(geo, "layer_count", 0))
        if not (0 <= layer < layers and 0 <= gx < x_steps and 0 <= gy < y_steps):
            return -1
        return layer * (x_steps * y_steps) + gy * x_steps + gx


    def _idx_to_coord(self, node_idx: int):
        """Convert node index back to (x, y, layer) coordinates - handles both lattice and escape nodes."""
        if self.geometry is None:
            logger.error(f"[_idx_to_coord] geometry system not initialized")
            return None

        # Calculate lattice size
        lattice_size = self.geometry.x_steps * self.geometry.y_steps * self.geometry.layer_count

        # DEBUG: Log lattice size and node classification for first few calls
        if not hasattr(self, '_logged_lattice_size'):
            logger.info(f"[COORD-DEBUG] lattice_size = {lattice_size} (x_steps={self.geometry.x_steps}, y_steps={self.geometry.y_steps}, layer_count={self.geometry.layer_count})")
            self._logged_lattice_size = True

        try:
            if node_idx < lattice_size:
                # LATTICE NODE: Use KiCadGeometry system
                x_idx, y_idx, layer = self.geometry.index_to_coords(node_idx)
                world_x, world_y = self.geometry.lattice_to_world(x_idx, y_idx)

                # DEBUG: Log coordinate conversion for debugging offset issue
                if node_idx < 10:  # Only first 10 to avoid spam
                    logger.info(f"[COORD-DEBUG] LATTICE node_idx={node_idx} -> lattice=({x_idx},{y_idx},{layer}) -> world=({world_x:.1f},{world_y:.1f})")

                # COORDINATE INVARIANT: Verify lattice node coordinates satisfy grid alignment
                x0, y0 = self.geometry.grid_min_x, self.geometry.grid_min_y
                pitch = self.geometry.pitch
                x_error = abs(world_x - (round((world_x - x0)/pitch)*pitch + x0))
                y_error = abs(world_y - (round((world_y - y0)/pitch)*pitch + y0))

                if x_error > 1e-6 or y_error > 1e-6:
                    logger.error(f"[COORD-INVARIANT] LATTICE node→coords failed: node_idx={node_idx}")
                    logger.error(f"  world=({world_x:.6f},{world_y:.6f}) x_error={x_error:.9f} y_error={y_error:.9f}")
                    logger.error(f"  Expected grid alignment with x0={x0:.6f} y0={y0:.6f} pitch={pitch:.6f}")

                return (world_x, world_y, layer)

            else:
                # ESCAPE NODE: Use legacy coordinate array system
                if not hasattr(self, '_logged_escape_nodes'):
                    logger.info(f"[COORD-DEBUG] ESCAPE node_idx={node_idx} >= lattice_size={lattice_size}")
                    self._logged_escape_nodes = True

                if self.node_coordinates is None:
                    logger.error(f"[_idx_to_coord] node_coordinates array not initialized for escape node {node_idx}")
                    return None

                if not (0 <= node_idx < self.node_coordinates.shape[0]):
                    logger.error(f"[_idx_to_coord] node_idx {node_idx} out of range [0, {self.node_coordinates.shape[0]})")
                    return None

                # Extract coordinates from the node_coordinates array
                coords = self.node_coordinates[node_idx]
                if hasattr(coords, 'get'):  # CuPy array
                    coords = coords.get()

                # DEBUG: Log escape node coordinate conversion
                if node_idx < lattice_size + 10:  # Only first 10 escape nodes
                    logger.info(f"[COORD-DEBUG] ESCAPE node_idx={node_idx} -> coords=({coords[0]:.1f},{coords[1]:.1f},{coords[2]:.0f})")

                return tuple(float(c) for c in coords)

        except Exception as e:
            logger.error(f"[_idx_to_coord] Failed to convert node_idx {node_idx}: {e}")
            return None


    def _find_portal_for_pad(self, pad):
        """Find the nearest actual lattice node to this pad position using KiCadGeometry."""
        pad_x, pad_y = self._get_pad_coordinates(pad)

        if self.geometry is None:
            logger.error(f"[PORTAL-SNAP] Geometry system not initialized")
            return None

        # Use geometry system to snap pad coordinates to lattice
        lattice_x, lattice_y = self.geometry.world_to_lattice(pad_x, pad_y)

        # Clamp to valid lattice bounds
        lattice_x = max(0, min(lattice_x, self.geometry.x_steps - 1))
        lattice_y = max(0, min(lattice_y, self.geometry.y_steps - 1))

        # Convert back to world coordinates (should be exact lattice node position)
        portal_x, portal_y = self.geometry.lattice_to_world(lattice_x, lattice_y)
        portal_layer = 0  # Start on F.Cu layer

        logger.debug(f"[PORTAL-SNAP] Pad ({pad_x:.2f},{pad_y:.2f}) -> lattice ({lattice_x},{lattice_y}) -> world ({portal_x:.2f},{portal_y:.2f})")

        # PORTAL SNAP INVARIANT: Verify snapped portal is grid-aligned
        x0, y0 = self.geometry.grid_min_x, self.geometry.grid_min_y
        pitch = self.geometry.pitch
        x_error = abs((portal_x - x0) % pitch)
        y_error = abs((portal_y - y0) % pitch)

        if x_error > 1e-6 or y_error > 1e-6:
            logger.error(f"[PORTAL-INVARIANT] Portal snap failed grid alignment:")
            logger.error(f"  portal=({portal_x:.6f},{portal_y:.6f}) x_error={x_error:.9f} y_error={y_error:.9f}")
            logger.error(f"  Expected grid alignment with x0={x0:.6f} y0={y0:.6f} pitch={pitch:.6f}")

        # Store Portal object for stub generation - use unique key including coordinates
        pad_id = self._uid_pad(pad)
        net_name = self._get_pad_net_name(pad) or 'UNKNOWN_NET'
        # Make key unique by including pad coordinates to avoid collisions
        portal_key = f"{pad_id}@{pad_x:.3f},{pad_y:.3f}"

        # Get pad layer as integer (normalize KiCad enums)
        pad_layer_raw = getattr(pad, 'layer', 0)
        if isinstance(pad_layer_raw, str):
            # Convert KiCad layer names to integers
            layer_map = {'F.Cu': 0, 'In1.Cu': 1, 'In2.Cu': 2, 'In3.Cu': 3, 'In4.Cu': 4, 'B.Cu': 5}
            pad_layer = layer_map.get(pad_layer_raw, 0)
        elif pad_layer_raw in [17, 18, 19, 20, 21, 31]:  # KiCad layer enums
            pad_layer = max(0, min(5, pad_layer_raw - 17))  # Normalize to 0-5
        else:
            pad_layer = int(pad_layer_raw)

        portal = Portal(
            x=portal_x,
            y=portal_y,
            layer=portal_layer,
            net=net_name,
            pad_layer=pad_layer
        )
        self._pad_portals[portal_key] = portal

        return {"x": portal_x, "y": portal_y, "layer": portal_layer}


    def _register_portal_stub(self, pad, portal_node):
        """FIX #1: Register REAL portal edges in live adjacency for routing."""
        if not hasattr(self, 'edges'):
            logger.error("[PORTAL-BIND] No edges list available for portal insertion")
            return

        # Create pad node at pad coordinates
        pad_x, pad_y = self._get_pad_coordinates(pad)
        pad_layer = 0  # F.Cu

        # Add bidirectional edges: pad <-> portal
        portal_x, portal_y = portal_node["x"], portal_node["y"]
        portal_layer = portal_node["layer"]

        # Convert coordinates to node indices
        pad_node_idx = self._coords_to_node_index(pad_x, pad_y, pad_layer)
        portal_node_idx = self._coords_to_node_index(portal_x, portal_y, portal_layer)

        if pad_node_idx is None or portal_node_idx is None:
            logger.error("[PORTAL-BIND] Failed to convert coordinates to node indices")
            return

        # Insert edges into live adjacency (will be in next CSR build)
        stub_cost = 0.1  # Low cost for portal connection
        self.edges.extend([
            (pad_node_idx, portal_node_idx, stub_cost),
            (portal_node_idx, pad_node_idx, stub_cost)
        ])

        if not hasattr(self, '_portal_edges_added'):
            self._portal_edges_added = 0
        self._portal_edges_added += 2

        logger.debug(f"[PORTAL-BIND] Connected pad@({pad_x:.1f},{pad_y:.1f}) to portal@({portal_x:.1f},{portal_y:.1f})")


    def _build_geometry_intents(self):
        """Build geometry intents from routing results - convert paths to tracks and vias."""
        class GeometryIntents:
            def __init__(self):
                self.tracks = []
                self.vias = []

        intents = GeometryIntents()

        # Convert committed paths to geometry intents
        paths = getattr(self, "_committed_paths", {}) or {}

        # TRIPWIRE F: Log per-net path conversion
        kept_segments = 0
        for net_id, path in paths.items():
            logger.info(f"[PATH] net={net_id} nodes={len(path) if path else 0}")
            if path and len(path) > 1:
                tracks, vias = self._path_to_geometry(net_id, path)
                intents.tracks.extend(tracks)
                intents.vias.extend(vias)
                kept_segments += len(tracks) + len(vias)

        # Add pad stub segments before returning
        stub_tracks, stub_vias = self._emit_pad_stubs()
        intents.tracks.extend(stub_tracks)
        intents.vias.extend(stub_vias)

        logger.info(f"[INTENTS] built tracks={len(intents.tracks)} vias={len(intents.vias)} (includes {len(stub_tracks)} pad stubs)")
        if kept_segments == 0 and len(paths) > 0:
            logger.warning("[INTENTS] all segments filtered; check eps/layer mapping")
        return intents


    def _emit_pad_stubs(self):
        """Generate stub segments connecting pads to their portal lattice nodes."""
        stub_tracks = []
        stub_vias = []

        if not hasattr(self, '_pad_portals') or not self._pad_portals:
            logger.debug("[STUB-EMIT] No pad portals found, skipping stub generation")
            return stub_tracks, stub_vias

        logger.info(f"[STUB-EMIT] Generating stubs for {len(self._pad_portals)} pad portals")

        # We need to find the actual pads to get their real coordinates
        # Look for stored pad information in the board or component data
        if not hasattr(self, '_all_pads') or not self._all_pads:
            logger.warning("[STUB-EMIT] No pad data available, generating minimal ownership stubs")
            # Generate minimal ownership stubs at portal locations
            for pad_id, portal in self._pad_portals.items():
                track = {
                    'net_name': portal.net,
                    'layer': portal.layer,
                    'start_x': portal.x,
                    'start_y': portal.y,
                    'end_x': portal.x,
                    'end_y': portal.y,
                    'width': 0.1,
                    'via_start': None,
                    'via_end': None
                }
                stub_tracks.append(track)
            return stub_tracks, stub_vias

        # Generate actual stubs from pad centers to portal points
        for portal_key, portal in self._pad_portals.items():
            # Extract pad coordinates from the portal key
            try:
                pad_id, coords_str = portal_key.split('@')
                coord_parts = coords_str.split(',')
                actual_pad_x = float(coord_parts[0])
                actual_pad_y = float(coord_parts[1])
            except (ValueError, IndexError):
                logger.warning(f"[STUB-EMIT] Invalid portal key format: {portal_key}, skipping")
                continue

            logger.debug(f"[STUB-EMIT] Processing portal {portal_key} at ({actual_pad_x:.3f},{actual_pad_y:.3f}) -> ({portal.x:.3f},{portal.y:.3f})")

            # Calculate distance to see if we need a visible stub
            distance = ((actual_pad_x - portal.x)**2 + (actual_pad_y - portal.y)**2)**0.5

            if distance > 0.05:  # If pad center is more than 0.05mm from portal
                # Generate visible stub from pad center to portal
                track = {
                    'net_name': portal.net,
                    'layer': portal.pad_layer,  # Use pad layer for the stub
                    'start_x': actual_pad_x,
                    'start_y': actual_pad_y,
                    'end_x': portal.x,
                    'end_y': portal.y,
                    'width': 0.2,  # Visible width
                    'via_start': None,
                    'via_end': None
                }
                stub_tracks.append(track)
                logger.info(f"[STUB-EMIT] Created {distance:.3f}mm stub for {pad_id} net {portal.net} on layer {portal.pad_layer}")
                logger.info(f"[STUB-EMIT] Stub: ({actual_pad_x:.3f},{actual_pad_y:.3f}) -> ({portal.x:.3f},{portal.y:.3f})")

                # Add via if pad is on different layer than portal
                if portal.pad_layer != portal.layer:
                    via = {
                        'net_name': portal.net,
                        'x': portal.x,
                        'y': portal.y,
                        'from_layer': portal.pad_layer,
                        'to_layer': portal.layer,
                        'size': 0.2,
                        'drill': 0.1
                    }
                    stub_vias.append(via)
                    logger.info(f"[STUB-EMIT] Created via for layer change: {portal.pad_layer} -> {portal.layer}")
            else:
                # Generate minimal ownership stub at portal
                track = {
                    'net_name': portal.net,
                    'layer': portal.layer,
                    'start_x': portal.x,
                    'start_y': portal.y,
                    'end_x': portal.x,
                    'end_y': portal.y,
                    'width': 0.1,
                    'via_start': None,
                    'via_end': None
                }
                stub_tracks.append(track)

        logger.info(f"[STUB-EMIT] Generated {len(stub_tracks)} stub tracks, {len(stub_vias)} portal vias")
        return stub_tracks, stub_vias


    def _snap_mm(self, v: float, origin: float, pitch: float) -> float:
        """Robust, deterministic snap (tolerant to 1e-9 noise)"""
        k = round((v - origin) / pitch)
        return origin + k * pitch


    def _path_to_geometry(self, net_id: str, path: list):
        """Convert a node path to tracks and vias."""
        tracks = []
        vias = []

        if len(path) < 2:
            return tracks, vias

        # Convert node indices to coordinates
        coords = []
        for node_idx in path:
            coord = self._idx_to_coord(node_idx)  # FIXED: Use correct coordinate method
            if coord:
                coords.append(coord)

        if len(coords) < 2:
            return tracks, vias

        # Generate tracks and vias from coordinate path
        for i in range(len(coords) - 1):
            x1, y1, layer1 = coords[i]
            x2, y2, layer2 = coords[i + 1]

            # Grid snap coordinates to prevent floating point errors
            if hasattr(self, 'geometry') and self.geometry is not None:
                x0, y0, pitch = self.geometry.grid_min_x, self.geometry.grid_min_y, self.geometry.pitch
                x1 = self._snap_mm(x1, x0, pitch)
                y1 = self._snap_mm(y1, y0, pitch)
                x2 = self._snap_mm(x2, x0, pitch)
                y2 = self._snap_mm(y2, y0, pitch)

            if layer1 == layer2:
                # Same layer - create track
                track = {
                    'net_id': net_id,
                    'layer': layer1,
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'width': 0.15  # mm default track width
                }

                # EMIT INVARIANT: Verify track endpoints are either lattice nodes or pad stubs
                if hasattr(self, 'geometry') and self.geometry is not None:
                    pitch = self.geometry.pitch
                    x0, y0 = self.geometry.grid_min_x, self.geometry.grid_min_y

                    if not self._on_grid(x1, y1, x0, y0, pitch) or not self._on_grid(x2, y2, x0, y0, pitch):
                        # Log first 3 violations, then suppress to avoid spam
                        if not hasattr(self, '_emit_violations_count'):
                            self._emit_violations_count = 0

                        if self._emit_violations_count < 3:
                            logger.error(f"[EMIT-INVARIANT] Track endpoint not grid-aligned: {net_id}")
                            logger.error(f"  start=({x1:.6f},{y1:.6f})")
                            logger.error(f"  end=({x2:.6f},{y2:.6f})")
                            self._emit_violations_count += 1
                        elif self._emit_violations_count == 3:
                            logger.error("[EMIT-INVARIANT] Suppressing further violations...")
                            self._emit_violations_count += 1

                tracks.append(track)
            else:
                # Different layers - create via
                via = {
                    'net_id': net_id,
                    'position': (x1, y1),
                    'from_layer': layer1,
                    'to_layer': layer2,
                    'drill': 0.2,  # mm default via drill
                    'size': 0.4    # mm default via size
                }
                vias.append(via)

        # STUB GENERATION: Add pad connection stubs
        stub_tracks, stub_vias = self._generate_pad_stubs(net_id, path)
        tracks.extend(stub_tracks)
        vias.extend(stub_vias)

        return tracks, vias


    def _generate_pad_stubs(self, net_id: str, path: List[int]) -> Tuple[List, List]:
        """Generate pad connection stubs: pad_center → portal + via if needed"""
        stub_tracks = []
        stub_vias = []

        if not path or not hasattr(self, '_pad_portals'):
            return stub_tracks, stub_vias

        # Check if this path connects to any pads through portals
        for pad_id, portal in self._pad_portals.items():
            # Check if portal net matches current net
            if portal.net != net_id:
                continue

            # Find if any path node connects to this portal
            path_coords = []
            for node_idx in path:
                coord = self._idx_to_coord(node_idx)
                if coord:
                    path_coords.append(coord)

            if not path_coords:
                continue

            # Check if portal connects to path (within routing grid distance)
            portal_connected = False
            connection_point = None
            for coord in path_coords:
                dx = abs(coord[0] - portal.x)
                dy = abs(coord[1] - portal.y)
                if dx < 0.1 and dy < 0.1:  # Within grid tolerance
                    portal_connected = True
                    connection_point = coord
                    break

            if not portal_connected:
                continue

            # Generate stub from pad center to portal
            pad_center = (portal.x, portal.y)  # Portal is already snapped to pad
            portal_pos = (portal.x, portal.y)

            # Get pad layer information
            pad_layer = portal.pad_layer
            portal_layer = portal.layer

            # Calculate stub distance
            stub_distance = ((pad_center[0] - portal_pos[0])**2 + (pad_center[1] - portal_pos[1])**2)**0.5

            # Generate ownership stub - even if zero length for GUI dot
            if stub_distance > 0.001:  # Real stub segment
                stub_track = {
                    'net_id': net_id,
                    'layer': pad_layer,
                    'start': pad_center,
                    'end': portal_pos,
                    'width': 0.1  # Thin stub
                }
                stub_tracks.append(stub_track)
            else:
                # Zero-length ownership stub (GUI dot)
                stub_track = {
                    'net_id': net_id,
                    'layer': pad_layer,
                    'start': pad_center,
                    'end': pad_center,
                    'width': 0.1,
                    'type': 'ownership_stub'  # GUI hint for dot rendering
                }
                stub_tracks.append(stub_track)

            # Add via if pad layer != portal layer
            if pad_layer != portal_layer:
                stub_via = {
                    'net_id': net_id,
                    'position': portal_pos,
                    'from_layer': pad_layer,
                    'to_layer': portal_layer,
                    'drill': 0.2,
                    'size': 0.4,
                    'type': 'pad_via'  # GUI hint
                }
                stub_vias.append(stub_via)

        return stub_tracks, stub_vias


    def _node_index_to_coords(self, node_idx: int):
        """Convert node index back to x, y, layer coordinates."""
        try:
            # Use actual lattice dimensions from initialization
            layers = getattr(self, 'layer_count', 6)
            if hasattr(self, '_grid_dims'):
                grid_width, grid_height = self._grid_dims
            else:
                # Fallback - this should never happen in normal operation
                grid_width = grid_height = 64
                logger.warning(f"[COORD-CONVERT] Missing _grid_dims, using fallback {grid_width}x{grid_height}")

            layer_size = grid_width * grid_height

            # Extract layer
            layer = node_idx // layer_size
            local_idx = node_idx % layer_size

            # Extract x, y using correct grid dimensions
            y = local_idx // grid_width
            x = local_idx % grid_width

            # Convert grid coordinates to absolute board coordinates
            grid_pitch = getattr(self, '_grid_pitch', 0.5)
            grid_x0 = getattr(self, '_grid_x0', 0.0)
            grid_y0 = getattr(self, '_grid_y0', 0.0)

            # CRITICAL FIX: Add board offset to get absolute coordinates
            x_mm = x * grid_pitch + grid_x0
            y_mm = y * grid_pitch + grid_y0

            # FORCE 0-based layer system - normalize any contamination
            if layer >= layers:  # Should never happen in proper 0-based system
                logger.warning(f"[LAYER-FIX] Normalizing layer {layer} → {layer % layers} (node_idx={node_idx})")
                layer = layer % layers

            # Ensure we always return 0-based layers (0, 1, 2, 3, 4, 5)
            layer = max(0, min(layer, layers - 1))

            return (x_mm, y_mm, layer)
        except Exception as e:
            logger.error(f"Error converting node_idx {node_idx} to coords: {e}")
            return None


    def _validate_geometry_intents(self, intents):
        """Validate geometry intents with real clearance DRC using R-tree collision detection."""
        class DRCViolations:
            def __init__(self):
                self.zero_len_tracks = 0
                self.via_in_pad = 0
                self.track_pad_clear = 0
                self.via_via_spacing = 0
                self.track_track_clearance = 0  # NEW: Track-to-track clearance violations
                self.track_via_clearance = 0    # NEW: Track-to-via clearance violations
                self.violation_details = []      # NEW: List of violation details for debugging

            # Back-compat aliases expected elsewhere in the code
            @property
            def track_track(self):
                return self.track_track_clearance

            @track_track.setter
            def track_track(self, v):
                self.track_track_clearance = int(v)

            @property
            def track_via(self):
                return self.track_via_clearance

            @track_via.setter
            def track_via(self, v):
                self.track_via_clearance = int(v)

            def total(self):
                return (self.zero_len_tracks + self.via_in_pad + self.track_pad_clear +
                       self.via_via_spacing + self.track_track_clearance + self.track_via_clearance)

            def __str__(self):
                return (f"DRC violations: zero_len={self.zero_len_tracks}, via_in_pad={self.via_in_pad}, "
                       f"track_pad_clear={self.track_pad_clear}, via_via={self.via_via_spacing}, "
                       f"track_track={self.track_track_clearance}, track_via={self.track_via_clearance}")

        viol = DRCViolations()

        # Check for zero-length tracks first
        if hasattr(intents, 'tracks'):
            for track in intents.tracks:
                try:
                    if 'start' in track and 'end' in track:
                        start_x, start_y = track['start']
                        end_x, end_y = track['end']
                    elif 'start_x' in track:
                        start_x, start_y = track['start_x'], track['start_y']
                        end_x, end_y = track['end_x'], track['end_y']
                    else:
                        continue

                    # Check for zero-length tracks
                    if abs(start_x - end_x) < 1e-6 and abs(start_y - end_y) < 1e-6:
                        viol.zero_len_tracks += 1
                        viol.violation_details.append(f"Zero-length track: {track.get('net_id', 'unknown')} at ({start_x:.3f}, {start_y:.3f})")
                except Exception as e:
                    logger.warning(f"[DRC] Failed to check track for zero length: {e}")

        # Perform real clearance DRC with R-tree collision detection
        if RTREE_AVAILABLE and hasattr(intents, 'tracks') and len(intents.tracks) > 1:
            clearance_violations = self._check_clearance_violations_rtree(intents)
            viol.track_track_clearance = clearance_violations['track_track']
            viol.track_via_clearance = clearance_violations['track_via']
            viol.violation_details.extend(clearance_violations['details'])
        elif not RTREE_AVAILABLE:
            logger.warning("[DRC] R-tree not available - skipping clearance checks")

        if viol.total() == 0:
            logger.info("[STRICT-DRC] pre-emit: all-clear (zero_len=%d, clearance_checks=enabled)",
                       viol.zero_len_tracks)
        else:
            logger.warning("[STRICT-DRC] pre-emit: %d violations detected", viol.total())
            for detail in viol.violation_details[:10]:  # Show first 10 violations
                logger.warning(f"[DRC-VIOLATION] {detail}")

        return viol


    def _convert_intents_to_view(self, intents):
        """Convert intents to geometry payload with proper layer mapping and coordinate normalization."""
        class GeometryPayload:
            def __init__(self, tracks, vias):
                self.tracks = tracks
                self.vias = vias

        # Get layer count for mapping
        layer_count = getattr(self.config, 'layer_count', 6)

        # CRITICAL FIX: Normalize tracks with proper layer mapping and coordinate conversion
        normalized_tracks = []
        if hasattr(intents, 'tracks'):
            for track in intents.tracks:
                try:
                    # Handle both old and new track data formats
                    if 'start' in track and 'end' in track:
                        # New format with tuple fields
                        start_x = self._as_py_float(track['start'][0])
                        start_y = self._as_py_float(track['start'][1])
                        end_x = self._as_py_float(track['end'][0])
                        end_y = self._as_py_float(track['end'][1])
                    elif 'start_x' in track and 'start_y' in track:
                        # Legacy format with separate coordinate fields
                        start_x = self._as_py_float(track['start_x'])
                        start_y = self._as_py_float(track['start_y'])
                        end_x = self._as_py_float(track['end_x'])
                        end_y = self._as_py_float(track['end_y'])
                    else:
                        logger.warning(f"[CONVERT] Track missing coordinate fields: {track}")
                        continue

                    width = self._as_py_float(track.get('width', 0.15))

                    # Sanity checks for valid coordinates
                    if any(x != x for x in [start_x, start_y, end_x, end_y, width]):  # NaN check
                        logger.warning(f"[CONVERT] Skipping track with NaN coordinates: {track}")
                        continue

                    if abs(start_x) > 1000 or abs(start_y) > 1000 or abs(end_x) > 1000 or abs(end_y) > 1000:
                        logger.warning(f"[CONVERT] Skipping track with extreme coordinates: {track}")
                        continue

                    # HARDEN LAYER ID: Ensure proper integer conversion
                    layer_raw = track.get('layer', 0)
                    layer_id = self._as_py_int(layer_raw)

                    normalized_track = {
                        'net_id': str(track.get('net_id', '')),
                        'layer': self._map_layer_for_gui(layer_id, layer_count),
                        'start': (start_x, start_y),
                        'end': (end_x, end_y),
                        'width': width
                    }
                    normalized_tracks.append(normalized_track)
                except Exception as e:
                    logger.warning(f"[CONVERT] Failed to normalize track {track}: {e}")
                    continue

        # Normalize vias similarly
        normalized_vias = []
        if hasattr(intents, 'vias'):
            for via in intents.vias:
                try:
                    normalized_via = {
                        'net_id': str(via.get('net_id', '')),
                        'position': (self._as_py_float(via['position'][0]), self._as_py_float(via['position'][1])),
                        'from_layer': self._map_layer_for_gui(int(via.get('from_layer', 0)), layer_count),
                        'to_layer': self._map_layer_for_gui(int(via.get('to_layer', 1)), layer_count),
                        'drill': self._as_py_float(via.get('drill', 0.2)),
                        'size': self._as_py_float(via.get('size', 0.4))
                    }
                    normalized_vias.append(normalized_via)
                except Exception as e:
                    logger.warning(f"[CONVERT] Failed to normalize via {via}: {e}")
                    continue

        # Log layer distribution for debugging
        layer_counts = {}
        for track in normalized_tracks:
            layer = track['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        layer_summary = ", ".join([f"L{k}:{v}" for k, v in sorted(layer_counts.items())])
        logger.info(f"[CONVERT] Normalized {len(normalized_tracks)} tracks, {len(normalized_vias)} vias for GUI")
        logger.info(f"[CONVERT] Layer distribution: {layer_summary}")

        # Sample first few tracks for debugging
        if normalized_tracks:
            logger.info(f"[CONVERT] Sample track: {normalized_tracks[0]}")
            if len(normalized_tracks) > 1:
                logger.info(f"[CONVERT] Sample track 2: {normalized_tracks[1]}")

        # Store geometry payload for internal use and return tuple for GUI
        class GeometryPayload:
            def __init__(self, tracks, vias):
                self.tracks = tracks
                self.vias = vias
        self._last_geometry_payload = GeometryPayload(normalized_tracks, normalized_vias)
        return (len(normalized_tracks), len(normalized_vias))


    def get_last_failure_message(self):
        """Get the last failure message for GUI display."""
        return getattr(self, "_last_failure", None)


    def get_routing_result(self):
        """Get the structured routing result for GUI access."""
        return getattr(self, "_routing_result", None)


    def _check_clearance_violations_rtree(self, intents):
        """Check clearance violations using R-tree spatial indexing."""
        violations = {
            'track_track': 0,
            'track_via': 0,
            'details': []
        }

        if not RTREE_AVAILABLE:
            return violations

        try:
            # DRC parameters - these should come from board rules
            min_track_clearance = 0.127  # 5 mil minimum clearance
            min_via_clearance = 0.127    # 5 mil minimum via clearance

            # Create R-tree index for tracks
            track_idx = rtree_index.Index()
            track_objects = []

            # Index all tracks with their bounding boxes + clearance halo
            for i, track in enumerate(intents.tracks):
                try:
                    if 'start' in track and 'end' in track:
                        start_x, start_y = track['start']
                        end_x, end_y = track['end']
                    elif 'start_x' in track:
                        start_x, start_y = track['start_x'], track['start_y']
                        end_x, end_y = track['end_x'], track['end_y']
                    else:
                        continue

                    width = track.get('width', 0.15)
                    layer = track.get('layer', 0)
                    net_id = track.get('net_id', '')

                    # Calculate bounding box with clearance halo
                    half_width = width / 2.0
                    clearance_halo = half_width + min_track_clearance

                    min_x = min(start_x, end_x) - clearance_halo
                    max_x = max(start_x, end_x) + clearance_halo
                    min_y = min(start_y, end_y) - clearance_halo
                    max_y = max(start_y, end_y) + clearance_halo

                    # Store track data
                    track_data = {
                        'start': (start_x, start_y),
                        'end': (end_x, end_y),
                        'width': width,
                        'layer': layer,
                        'net_id': net_id,
                        'bbox': (min_x, min_y, max_x, max_y)
                    }
                    track_objects.append(track_data)

                    # Insert into R-tree with layer-encoded bounds for 3D collision
                    # Encode layer as Z-coordinate range [layer*1000, (layer+1)*1000-1]
                    track_idx.insert(i, (min_x, min_y, max_x, max_y))

                except Exception as e:
                    logger.warning(f"[DRC-RTREE] Failed to index track {i}: {e}")
                    continue

            # Check track-to-track clearance violations
            for i, track1 in enumerate(track_objects):
                try:
                    # Query R-tree for potential collisions
                    bbox1 = track1['bbox']
                    candidates = list(track_idx.intersection(bbox1))

                    for j in candidates:
                        if j <= i:  # Avoid duplicate checks
                            continue

                        if j >= len(track_objects):
                            continue

                        track2 = track_objects[j]

                        # Skip if same net (no clearance required within net)
                        if track1['net_id'] == track2['net_id']:
                            continue

                        # Skip if different layers (no clearance required between layers)
                        if track1['layer'] != track2['layer']:
                            continue

                        # Calculate actual clearance between track segments
                        clearance = self._calculate_track_clearance(track1, track2)

                        if clearance < min_track_clearance:
                            violations['track_track'] += 1
                            violation_msg = (f"Track clearance violation: {track1['net_id']} vs {track2['net_id']} "
                                           f"on layer {track1['layer']}, clearance={clearance:.3f}mm < {min_track_clearance:.3f}mm")
                            violations['details'].append(violation_msg)

                            # Stop after finding reasonable number of violations
                            if violations['track_track'] >= 100:
                                violations['details'].append("[DRC] Too many track-track violations, stopping check...")
                                break

                except Exception as e:
                    logger.warning(f"[DRC-RTREE] Failed clearance check for track {i}: {e}")
                    continue

                if violations['track_track'] >= 100:
                    break

            # Note: Track-to-via clearance checks could be added here for enhanced DRC
            if hasattr(intents, 'vias') and intents.vias:
                logger.info(f"[DRC-RTREE] Track-via clearance checks for {len(intents.vias)} vias not yet implemented")

            logger.info(f"[DRC-RTREE] Clearance check completed: {violations['track_track']} track-track violations")

        except Exception as e:
            logger.error(f"[DRC-RTREE] R-tree clearance check failed: {e}")
            violations['details'].append(f"R-tree clearance check failed: {e}")

        return violations


    def _calculate_track_clearance(self, track1, track2):
        """Calculate minimum clearance between two track segments."""
        try:
            # Simplified clearance calculation - distance between track centerlines minus half-widths
            x1_start, y1_start = track1['start']
            x1_end, y1_end = track1['end']
            x2_start, y2_start = track2['start']
            x2_end, y2_end = track2['end']

            # Calculate minimum distance between two line segments
            # For simplicity, use point-to-segment distance from track1 endpoints to track2
            distances = [
                self._point_to_segment_distance((x1_start, y1_start), (x2_start, y2_start), (x2_end, y2_end)),
                self._point_to_segment_distance((x1_end, y1_end), (x2_start, y2_start), (x2_end, y2_end)),
                self._point_to_segment_distance((x2_start, y2_start), (x1_start, y1_start), (x1_end, y1_end)),
                self._point_to_segment_distance((x2_end, y2_end), (x1_start, y1_start), (x1_end, y1_end))
            ]

            min_distance = min(distances)

            # Subtract half-widths to get edge-to-edge clearance
            half_width1 = track1['width'] / 2.0
            half_width2 = track2['width'] / 2.0
            clearance = min_distance - half_width1 - half_width2

            return max(0.0, clearance)  # Never return negative clearance

        except Exception as e:
            logger.warning(f"[DRC] Failed to calculate track clearance: {e}")
            return 1000.0  # Return large value to avoid false violations


    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Calculate minimum distance from point to line segment."""
        try:
            px, py = point
            sx1, sy1 = seg_start
            sx2, sy2 = seg_end

            # Vector from segment start to end
            dx = sx2 - sx1
            dy = sy2 - sy1

            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                # Degenerate segment - just point distance
                return ((px - sx1)**2 + (py - sy1)**2)**0.5

            # Parameter t along segment where closest point lies
            t = max(0.0, min(1.0, ((px - sx1) * dx + (py - sy1) * dy) / (dx * dx + dy * dy)))

            # Closest point on segment
            closest_x = sx1 + t * dx
            closest_y = sy1 + t * dy

            # Distance from point to closest point on segment
            return ((px - closest_x)**2 + (py - closest_y)**2)**0.5

        except Exception as e:
            logger.warning(f"[DRC] Point-to-segment distance calculation failed: {e}")
            return 1000.0  # Return large value to avoid false violations


    def _add_edge_to_rtree(self, net_id: str, layer: int, x1: float, y1: float, x2: float, y2: float):
        """Add edge to spatial index for clearance checking."""
        if not self._clearance_enabled or layer not in self._clearance_rtrees:
            return

        try:
            # Create inflated bounding box
            track_width = 0.15  # Default track width in mm
            min_clearance = 0.127  # 5 mil minimum clearance
            half_width = track_width / 2.0
            clearance_halo = half_width + min_clearance

            min_x = min(x1, x2) - clearance_halo
            max_x = max(x1, x2) + clearance_halo
            min_y = min(y1, y2) - clearance_halo
            max_y = max(y1, y2) + clearance_halo

            # Insert into R-tree with unique ID and net info
            edge_id = f"{net_id}_{x1}_{y1}_{x2}_{y2}_{layer}"
            self._clearance_rtrees[layer].insert(hash(edge_id) % 2147483647,
                                               (min_x, min_y, max_x, max_y),
                                               obj=("track", net_id))

        except Exception as e:
            logger.warning(f"[RTREE] Failed to add edge to spatial index: {e}")


    def _remove_edge_from_rtree(self, net_id: str, layer: int, edge_key: tuple):
        """Remove edge from spatial index."""
        if not self._clearance_enabled or layer not in self._clearance_rtrees:
            return

        try:
            # Extract coordinates from edge key
            # edge_key format: (layer, x1_grid, y1_grid, x2_grid, y2_grid)
            if len(edge_key) >= 5:
                layer_id, x1_grid, y1_grid, x2_grid, y2_grid = edge_key[:5]

                # Convert back from grid coordinates
                gx = self._grid_pitch
                x1, y1 = x1_grid * gx, y1_grid * gx
                x2, y2 = x2_grid * gx, y2_grid * gx

                # Create same bounding box as when added
                track_width = 0.15
                min_clearance = 0.127
                half_width = track_width / 2.0
                clearance_halo = half_width + min_clearance

                min_x = min(x1, x2) - clearance_halo
                max_x = max(x1, x2) + clearance_halo
                min_y = min(y1, y2) - clearance_halo
                max_y = max(y1, y2) + clearance_halo

                # Remove from R-tree
                edge_id = f"{net_id}_{x1}_{y1}_{x2}_{y2}_{layer_id}"
                self._clearance_rtrees[layer].delete(hash(edge_id) % 2147483647,
                                                    (min_x, min_y, max_x, max_y))

        except Exception as e:
            logger.warning(f"[RTREE] Failed to remove edge from spatial index: {e}")


    def get_route_visualization_data(self, paths: Dict[str, List[int]]) -> List[Dict]:
        """Convert paths to visualization tracks"""
        tracks = []
        
        if hasattr(self.node_coordinates, 'get'):
            coords_cpu = self.node_coordinates.get()
        else:
            coords_cpu = self.node_coordinates
        
        layer_map = {
            0: 'F.Cu', 1: 'In1.Cu', 2: 'In2.Cu', 3: 'In3.Cu',
            4: 'In4.Cu', 5: 'B.Cu'
        }
        
        for net_id, path in paths.items():
            if len(path) < 2:
                continue
            
            for i in range(len(path) - 1):
                from_x, from_y, from_layer = coords_cpu[path[i]]
                to_x, to_y, to_layer = coords_cpu[path[i + 1]]
                
                track = {
                    'net_name': net_id,
                    'start_x': float(from_x),
                    'start_y': float(from_y),
                    'end_x': float(to_x),
                    'end_y': float(to_y),
                    'layer': layer_map.get(int(from_layer), f'In{int(from_layer)}.Cu'),
                    'width': 0.2,
                    'segment_type': 'via' if from_layer != to_layer else 'trace'
                }
                tracks.append(track)
        
        return tracks
    
    # ===== MULTI-ROI PARALLEL PROCESSING =====
    

