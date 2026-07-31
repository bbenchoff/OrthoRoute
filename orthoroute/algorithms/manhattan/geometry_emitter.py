"""
Geometry emission for PathFinderRouter.

Extracted verbatim from unified_pathfinder.py (hardening audit, Phase 8
extraction 1). GeometryEmitter is a collaborator that operates on the router
instance passed to its constructor; PathFinderRouter keeps thin delegating
methods so every internal and external call site is unchanged.
"""

import logging
from typing import List, Tuple

import numpy as np

from ...domain.models.board import Board
from .hdi_stack import canonical_pair

logger = logging.getLogger(__name__)


class GeometryPayload:
    """Wrapper for geometry with attribute access"""
    def __init__(self, tracks, vias):
        self.tracks = tracks
        self.vias = vias


class GeometryEmitter:
    """Emits final/provisional track+via geometry from routed paths.

    Holds no state of its own beyond the router reference: all reads and
    writes go through the router so behavior is identical to the
    pre-extraction methods.
    """

    def __init__(self, router):
        self._router = router

    def _segment_world(self, a_idx: int, b_idx: int, layer: int, net: str):
        ax, ay, _ = self._router.lattice.idx_to_coord(a_idx)
        bx, by, _ = self._router.lattice.idx_to_coord(b_idx)
        (ax_mm, ay_mm) = self._router.lattice.geom.lattice_to_world(ax, ay)
        (bx_mm, by_mm) = self._router.lattice.geom.lattice_to_world(bx, by)

        # QUANTIZE: Round to grid to prevent float drift
        pitch = self._router.lattice.geom.pitch
        origin_x = self._router.lattice.geom.grid_min_x
        origin_y = self._router.lattice.geom.grid_min_y

        ax_mm = origin_x + round((ax_mm - origin_x) / pitch) * pitch
        ay_mm = origin_y + round((ay_mm - origin_y) / pitch) * pitch
        bx_mm = origin_x + round((bx_mm - origin_x) / pitch) * pitch
        by_mm = origin_y + round((by_mm - origin_y) / pitch) * pitch

        return {
            'net': net,
            'layer': self._router.config.layer_names[layer] if layer < len(self._router.config.layer_names) else f"L{layer}",
            'x1': ax_mm, 'y1': ay_mm, 'x2': bx_mm, 'y2': by_mm,
            'width': self._router.config.track_width,
        }

    def _via_world(self, at_idx: int, net: str, from_layer: int, to_layer: int):
        x, y, _ = self._router.lattice.idx_to_coord(at_idx)
        (x_mm, y_mm) = self._router.lattice.geom.lattice_to_world(x, y)

        # CRITICAL FIX: Quantize via coordinates to grid (same as _segment_world)
        # This ensures via centers EXACTLY match track endpoints (no epsilon mismatch!)
        pitch = self._router.lattice.geom.pitch
        origin_x = self._router.lattice.geom.grid_min_x
        origin_y = self._router.lattice.geom.grid_min_y
        x_mm = origin_x + round((x_mm - origin_x) / pitch) * pitch
        y_mm = origin_y + round((y_mm - origin_y) / pitch) * pitch

        # Normalize layer order (consistent output, KiCad accepts either way)
        if from_layer > to_layer:
            from_layer, to_layer = to_layer, from_layer

        via = {
            'net': net,
            'x': x_mm, 'y': y_mm,
            'from_layer': self._router.config.layer_names[from_layer] if from_layer < len(self._router.config.layer_names) else f"L{from_layer}",
            'to_layer': self._router.config.layer_names[to_layer] if to_layer < len(self._router.config.layer_names) else f"L{to_layer}",
            'diameter': self._router.config.via_diameter,
            'drill': self._router.config.via_drill,
        }
        hdi_stack = getattr(self._router.config, "hdi_stack", None)
        if hdi_stack is not None:
            process = hdi_stack.process_for_span(
                from_layer, to_layer
            )
            via.update({
                "diameter": process.diameter_mm,
                "drill": process.drill_mm,
                "via_process": process.name,
                "via_kind": process.kind,
                "hdi_stack": hdi_stack.name,
            })
        return via

    def _expand_hdi_vias(self, vias: List[dict]) -> List[dict]:
        """Express every emitted HDI transition as legal physical spans."""
        hdi_stack = getattr(self._router.config, "hdi_stack", None)
        if hdi_stack is None:
            return list(vias)

        expanded = []
        for via in vias:
            from_layer = self._router._layer_name_to_index(
                via.get("from_layer")
            )
            to_layer = self._router._layer_name_to_index(
                via.get("to_layer")
            )
            if from_layer is None or to_layer is None:
                raise ValueError(
                    "HDI via has an unknown layer span: "
                    f"{via.get('from_layer')} -> {via.get('to_layer')}"
                )
            for physical_from, physical_to in hdi_stack.expand_span(
                from_layer, to_layer
            ):
                lo, hi = canonical_pair(physical_from, physical_to)
                process = hdi_stack.process_for_span(lo, hi)
                item = dict(via)
                item.update({
                    "from_layer": self._router.config.layer_names[lo],
                    "to_layer": self._router.config.layer_names[hi],
                    "diameter": process.diameter_mm,
                    "drill": process.drill_mm,
                    "via_process": process.name,
                    "via_kind": process.kind,
                    "hdi_stack": hdi_stack.name,
                })
                expanded.append(item)
        return expanded

    def _refresh_selected_escape_geometry(self) -> None:
        """Emit stubs only for the portal candidates selected by routing."""
        if self._router.escape_planner is None:
            return

        tracks = []
        vias = []
        emitted_pads = set()
        for net_id, pad_ids in self._router.net_pad_ids.items():
            if not self._router.net_paths.get(net_id):
                continue
            selected = self._router.net_selected_portals.get(net_id)
            if selected is None:
                selected = tuple(
                    self._router.portals.get(pad_id) for pad_id in pad_ids
                )
            layers = self._router.net_portal_layers.get(net_id, (1, 1))

            for pad_id, portal, entry_layer in zip(
                pad_ids, selected, layers
            ):
                if portal is None or pad_id in emitted_pads:
                    continue
                emitted_pads.add(pad_id)
                geometry = (
                    self._router.escape_planner._emit_portal_escape_geometry(
                        net_id,
                        pad_id,
                        portal,
                        entry_layer,
                        include_via=True,
                    )
                )
                for item in geometry:
                    if "x1" in item and "y1" in item:
                        tracks.append(item)
                    elif "x" in item and "y" in item:
                        vias.append(item)

        self._router._escape_tracks = tracks
        self._router._escape_vias = vias

    def emit_geometry(self, board: Board) -> Tuple[int, int]:
        """
        Convert routed node paths into drawable segments and vias.
        - Clean geometry (for KiCad export): only if overuse == 0
        - Provisional geometry (for GUI feedback): always generated

        CRITICAL: Escape geometry is ALWAYS merged, even with overuse.
        Escapes are the connection from pads to the routing grid and must be exported.
        """
        self._router._refresh_selected_escape_geometry()

        # Generate provisional geometry from routing paths
        provisional_tracks, provisional_vias = self._router._generate_geometry_from_paths()

        # ALWAYS merge escape geometry with routed geometry
        # Deduplicate helper
        def _dedupe(items, key_fn):
            seen, out = set(), []
            for it in items:
                k = key_fn(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
            return out

        final_tracks = provisional_tracks
        final_vias = provisional_vias

        if hasattr(self._router, '_escape_tracks') and self._router._escape_tracks:
            # Merge escapes first (so they're visually "underneath")
            combined_tracks = self._router._escape_tracks + provisional_tracks
            combined_vias = self._router._escape_vias + provisional_vias

            # Deduplicate by geometric signature
            final_tracks = _dedupe(
                combined_tracks,
                lambda t: (t["net"], t["layer"],
                          round(t["x1"], 3), round(t["y1"], 3),
                          round(t["x2"], 3), round(t["y2"], 3),
                          round(t["width"], 3))
            )
            final_vias = _dedupe(
                combined_vias,
                lambda v: (v["net"], round(v["x"], 3), round(v["y"], 3),
                          v.get("from_layer"), v.get("to_layer"),
                          round(v.get("drill", 0), 3),
                          round(v.get("diameter", 0), 3))
            )

            logger.info(f"[ESCAPE-MERGE] escapes={len(self._router._escape_tracks)} + "
                       f"routed={len(provisional_tracks)} → "
                       f"total={len(final_tracks)} tracks after dedup")
            logger.info(f"[ESCAPE-MERGE] escape_vias={len(self._router._escape_vias)} + "
                       f"routed_vias={len(provisional_vias)} → "
                       f"total={len(final_vias)} vias after dedup")

        final_vias = self._router._expand_hdi_vias(final_vias)
        if getattr(self._router.config, "hdi_stack", None) is not None:
            final_vias = _dedupe(
                final_vias,
                lambda v: (
                    v["net"],
                    round(v["x"], 3),
                    round(v["y"], 3),
                    v.get("from_layer"),
                    v.get("to_layer"),
                    round(v.get("drill", 0), 4),
                    round(v.get("diameter", 0), 4),
                    v.get("via_process"),
                ),
            )

        # Store merged geometry as provisional (for GUI display)
        self._router._provisional_geometry = GeometryPayload(final_tracks, final_vias)

        # Check for overuse (include via spatial violations)
        over_sum, over_cnt = self._router.accounting.compute_overuse(router_instance=self._router)

        if over_sum > 0:
            logger.warning(f"[EMIT] Overuse={over_sum}: showing merged geometry in GUI but not exporting to KiCad")
            self._router._geometry_payload = GeometryPayload([], [])  # No clean geometry for export
            # Return merged counts so GUI shows escapes + routes
            return (len(final_tracks), len(final_vias))

        # No overuse: emit clean geometry for KiCad export
        logger.info("[EMIT] Routing converged! Exporting clean geometry with escapes")
        self._router._geometry_payload = GeometryPayload(final_tracks, final_vias)
        return (len(final_tracks), len(final_vias))

    def _path_without_dynamic_escape_chains(
        self, net_id: str, path: List[int]
    ) -> List[int]:
        """Remove terminal barrels supplied by explicit escape geometry."""
        selected = self._router.net_selected_portals.get(net_id)
        layers = self._router.net_portal_layers.get(net_id)
        if not path or selected is None or layers is None:
            return list(path)

        start = 0
        end = len(path)
        src_portal, dst_portal = selected
        src_layer, dst_layer = layers
        target = (
            src_portal.x_idx,
            src_portal.y_idx,
            src_layer,
        )
        for index, node in enumerate(path):
            if self._router.lattice.idx_to_coord(node) == target:
                start = index
                break
        target = (
            dst_portal.x_idx,
            dst_portal.y_idx,
            dst_layer,
        )
        for index in range(len(path) - 1, start - 1, -1):
            if self._router.lattice.idx_to_coord(path[index]) == target:
                end = index + 1
                break
        return list(path[start:end])

    def _generate_geometry_from_paths(self) -> Tuple[List, List]:
        """Generate tracks and vias from net_paths"""
        tracks, vias = [], []

        for net_id, path in self._router.net_paths.items():
            if not path:
                continue
            path = self._router._path_without_dynamic_escape_chains(
                net_id, path
            )
            if not path:
                continue
            if getattr(self._router.config, "hdi_stack", None) is None:
                path = self._router._coalesce_vertical_runs(path)

            # NOTE: Escape geometry is pre-computed by PadEscapePlanner and cached.
            # It will be merged with routed geometry in emit_geometry().

            # Generate tracks/vias from main path
            run_start = path[0]
            prev = path[0]
            prev_dir = None
            prev_layer = self._router.lattice.idx_to_coord(prev)[2]

            for node in path[1:]:
                x0, y0, z0 = self._router.lattice.idx_to_coord(prev)
                x1, y1, z1 = self._router.lattice.idx_to_coord(node)

                # Drop any planar segment on outer layers (shouldn't happen once graph/ROI are fixed)
                if z0 == z1 and (z0 == 0 or z0 == self._router.lattice.layers - 1):
                    logger.error(f"[EMIT-GUARD] refusing planar segment on outer layer {z0} for net {net_id}")
                    prev = node
                    prev_layer = z1
                    run_start = node
                    continue

                # VALIDATION: Check if nodes are adjacent (Manhattan distance should be 1)
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                dz = abs(z1 - z0)

                if dz == 0:  # Same layer - enforce H/V discipline
                    # Must be adjacent
                    if (dx + dy) != 1:
                        logger.error(f"[GEOMETRY-BUG] Non-adjacent nodes in path for net {net_id}: "
                                   f"({x0},{y0},{z0}) → ({x1},{y1},{z1}), Manhattan dist = {dx+dy}")
                        logger.error(f"[GEOMETRY-BUG] Path indices: prev={prev}, node={node}")
                        logger.error(f"[GEOMETRY-BUG] This creates diagonal segment! GPU parent pointers are CORRUPT!")
                        continue  # Skip illegal segment

                    # Check layer direction discipline
                    layer_axis = "h" if dy == 0 else "v"
                    if layer_axis not in self._router.lattice.get_allowed_axes(z0):
                        logger.error(
                            "[LAYER-VIOLATION] Axis %s is unavailable "
                            "on layer %d",
                            layer_axis,
                            z0,
                        )
                        continue
                    if layer_axis == 'h':
                        # H layer: y must be constant (horizontal movement)
                        if dy != 0:
                            logger.error(f"[LAYER-VIOLATION] H-layer {z0} has vertical move: "
                                       f"({x0},{y0})→({x1},{y1}), dy={dy}")
                            continue
                    else:  # 'v'
                        # V layer: x must be constant (vertical movement)
                        if dx != 0:
                            logger.error(f"[LAYER-VIOLATION] V-layer {z0} has horizontal move: "
                                       f"({x0},{y0})→({x1},{y1}), dx={dx}")
                            continue

                if z1 != z0:
                    # flush any pending straight run before via
                    if prev != run_start:
                        tracks.append(self._router._segment_world(run_start, prev, prev_layer, net_id))
                    vias.append(self._router._via_world(prev, net_id, z0, z1))
                    run_start = node
                    prev_dir = None
                else:
                    dir_vec = (np.sign(x1 - x0), np.sign(y1 - y0))
                    if prev_dir is None or dir_vec == prev_dir:
                        # keep extending run
                        pass
                    else:
                        # direction changed: flush previous run
                        tracks.append(self._router._segment_world(run_start, prev, prev_layer, net_id))
                        run_start = prev
                    prev_dir = dir_vec

                prev = node
                prev_layer = z1

            # flush final run
            if prev != run_start:
                tracks.append(self._router._segment_world(run_start, prev, prev_layer, net_id))

        # FINAL VALIDATION: Check all tracks are axis-aligned
        violations = []
        for i, track in enumerate(tracks):
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            # Must be axis-aligned (one coordinate must be constant)
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 0.001 and dy > 0.001:
                violations.append((i, track, dx, dy))

        if violations:
            logger.error(f"[EMIT-VALIDATION] Found {len(violations)} diagonal segments!")
            for i, track, dx, dy in violations[:5]:  # Show first 5
                logger.error(f"  Track {i}: ({track['x1']:.2f},{track['y1']:.2f})->({track['x2']:.2f},{track['y2']:.2f}), "
                           f"Delta=({dx:.2f},{dy:.2f}) on {track['layer']}")

            # In debug mode, raise error
            if __debug__:
                raise RuntimeError(f"{len(violations)} diagonal segments detected at emission")
        else:
            logger.info(f"[EMIT-VALIDATION] All {len(tracks)} tracks are axis-aligned ✓")

        # Count tracks by layer and direction
        layer_stats = {}
        for track in tracks:
            layer = track['layer']
            x1, y1 = track['x1'], track['y1']
            x2, y2 = track['x2'], track['y2']

            is_horizontal = (abs(y1 - y2) < 0.001)
            is_vertical = (abs(x1 - x2) < 0.001)

            if layer not in layer_stats:
                layer_stats[layer] = {'h': 0, 'v': 0}

            if is_horizontal:
                layer_stats[layer]['h'] += 1
            elif is_vertical:
                layer_stats[layer]['v'] += 1

        # Log per-layer statistics and check direction discipline
        for layer in sorted(layer_stats.keys()):
            h_count = layer_stats[layer]['h']
            v_count = layer_stats[layer]['v']
            logger.info(f"[LAYER-STATS] {layer}: {h_count} horizontal, {v_count} vertical")

            try:
                layer_index = self._router.config.layer_names.index(layer)
                expected_dir = self._router.lattice.get_legal_axis(layer_index)
                if expected_dir == 'h' and v_count > h_count:
                    logger.warning(
                        f"[LAYER-DIRECTION] {layer} is H-preferred "
                        "but has more V traces"
                    )
                elif expected_dir == 'v' and h_count > v_count:
                    logger.warning(
                        f"[LAYER-DIRECTION] {layer} is V-preferred "
                        "but has more H traces"
                    )
            except (ValueError, IndexError):
                pass

        return (tracks, vias)

    def _coalesce_vertical_runs(self, path: List[int]) -> List[int]:
        """Collapse adjacent z hops at one x/y into one physical via span."""
        if len(path) < 3:
            return list(path)

        result = [path[0]]
        via_xy = None
        via_direction = 0

        for previous, node in zip(path, path[1:]):
            x0, y0, z0 = self._router.lattice.idx_to_coord(previous)
            x1, y1, z1 = self._router.lattice.idx_to_coord(node)
            dz = z1 - z0
            is_vertical = x0 == x1 and y0 == y1 and dz != 0
            direction = int(np.sign(dz)) if is_vertical else 0

            if (
                is_vertical
                and via_xy == (x0, y0)
                and direction == via_direction
            ):
                result[-1] = node
            else:
                result.append(node)

            if is_vertical:
                via_xy = (x0, y0)
                via_direction = direction
            else:
                via_xy = None
                via_direction = 0

        return result

    def get_geometry_payload(self):
        """
        Get geometry payload for GUI/export.

        Returns clean geometry if available (no overuse),
        otherwise returns provisional geometry so GUI can still display/export.
        """
        # If clean geometry is empty but provisional exists, return provisional
        if (not self._router._geometry_payload.tracks and not self._router._geometry_payload.vias
            and hasattr(self._router, '_provisional_geometry')
            and (self._router._provisional_geometry.tracks or self._router._provisional_geometry.vias)):
            return self._router._provisional_geometry
        return self._router._geometry_payload

    def get_provisional_geometry(self):
        """Get provisional geometry for GUI feedback (always available)"""
        return self._router._provisional_geometry
