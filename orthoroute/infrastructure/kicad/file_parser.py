"""KiCad board file parser (direct .kicad_pcb parsing, no KiCad needed).

Built on the balanced-paren s-expression parser in ``sexpr.py`` and
supports every dialect from KiCad 5 through KiCad 10:

- KiCad <= 7: numbered net table ``(net 3 "GND")``, references in
  ``(fp_text reference ...)``, pads carry ``(net 3 "GND")``.
- KiCad 8/9: references move to ``(property "Reference" ...)``.
- KiCad 10 (format 20260206): the net table is gone; nets exist only as
  name references ``(net "GND")`` on pads/tracks/zones and are synthesized
  here from pad references.

Net ids are normalized to the net *name* for both dialects so downstream
code never sees dialect-specific numbering.

Pad positions are stored in the footprint's local frame; this parser
composes them with the footprint position/rotation (and back-side mirror)
to produce world coordinates — the legacy regex parser never did, so pad
world positions were wrong for any rotated footprint.

Python 3.9 compatible (imported by plugin runtime code).
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models.board import Board, Component, Coordinate, Layer, Net, Pad
from .sexpr import SExpr, atoms, child, children, first_atom, node_name, parse

logger = logging.getLogger(__name__)


def _f(value: Optional[str], default: float = 0.0) -> float:
    """Parse a float atom with a default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class KiCadFileParser:
    """Parser for KiCad board files (.kicad_pcb)."""

    def load_board(self, file_path: str) -> Optional[Board]:
        """Load board from KiCad file."""
        try:
            board_data = self.parse_file(file_path)
            return self.create_board_from_data(board_data)
        except Exception as e:
            logger.error(f"Failed to load board from {file_path}: {e}")
            return None

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a KiCad board file into a plain-dict description."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Board file not found: {file_path}")
        if path.suffix.lower() != ".kicad_pcb":
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return self._parse_kicad_pcb(path)

    def _parse_kicad_pcb(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        doc = parse(text)
        if not doc or node_name(doc[0]) != "kicad_pcb":
            raise ValueError(f"Not a kicad_pcb file: {path}")
        root = doc[0]

        version = int(_f(first_atom(child(root, "version")), 0))
        net_table = self._extract_net_table(root)
        components = self._extract_components(root, net_table)
        nets = self._collect_nets(net_table, components)

        board_data = {
            "title": self._extract_title(root) or path.stem,
            "version": version,
            "layers": self._extract_layers(root),
            "components": components,
            "nets": nets,
            "design_rules": self._extract_design_rules(root, path),
            "tracks": self._extract_tracks(root, net_table),
            "vias": self._extract_vias(root, net_table),
        }
        logger.info(
            f"Parsed {path.name} (format {version}): "
            f"{len(components)} components, {len(nets)} nets, "
            f"{sum(len(c['pads']) for c in components)} pads"
        )
        return board_data

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(root: SExpr) -> Optional[str]:
        title_block = child(root, "title_block")
        if title_block is not None:
            return first_atom(child(title_block, "title"))
        return None

    @staticmethod
    def _extract_layers(root: SExpr) -> List[Dict[str, Any]]:
        """Board layer table: entries are (<id> "<name>" <type> ["<user name>"])."""
        layers_node = child(root, "layers")
        if layers_node is None:
            return [
                {"id": 0, "name": "F.Cu", "type": "signal", "stackup_position": 0},
                {"id": 2, "name": "B.Cu", "type": "signal", "stackup_position": 2},
            ]
        layers = []
        for entry in layers_node[1:]:
            if not isinstance(entry, list) or len(entry) < 3:
                continue
            layers.append({
                "id": int(_f(entry[0], -1)),
                "name": entry[1],
                "type": entry[2],
                "stackup_position": int(_f(entry[0], -1)),
            })
        return layers

    @staticmethod
    def _extract_net_table(root: SExpr) -> Dict[str, str]:
        """Legacy numbered net table: {net_code: net_name}. Empty on KiCad 10."""
        table: Dict[str, str] = {}
        for net in children(root, "net"):
            vals = atoms(net)
            if len(vals) >= 2 and vals[1]:
                table[vals[0]] = vals[1]
        return table

    @staticmethod
    def _pad_net_name(pad_node: SExpr, net_table: Dict[str, str]) -> Optional[str]:
        """Net name from a (net ...) child: numbered (legacy) or name-only (10)."""
        net = child(pad_node, "net")
        if net is None:
            return None
        vals = atoms(net)
        if not vals:
            return None
        if len(vals) >= 2:  # (net <code> "NAME")
            return vals[1] or None
        # (net "NAME") — KiCad 10; or (net <code>) — resolve via table
        val = vals[0]
        if val in net_table:
            return net_table[val] or None
        return val or None

    def _extract_components(self, root: SExpr,
                            net_table: Dict[str, str]) -> List[Dict[str, Any]]:
        components = []
        for fp in children(root, "footprint") + children(root, "module"):
            footprint_lib = first_atom(fp) or ""

            at = child(fp, "at")
            at_vals = atoms(at) if at is not None else []
            fx = _f(at_vals[0] if len(at_vals) > 0 else None)
            fy = _f(at_vals[1] if len(at_vals) > 1 else None)
            fangle = _f(at_vals[2] if len(at_vals) > 2 else None)

            layer = first_atom(child(fp, "layer")) or "F.Cu"
            back = layer.startswith("B.")

            reference = self._extract_reference(fp)
            value = self._extract_value(fp)

            component = {
                "footprint": footprint_lib,
                "reference": reference,
                "value": value,
                "x": fx, "y": fy,
                "angle": fangle,
                "layer": layer,
                "pads": self._extract_pads(fp, reference, fx, fy, fangle, back,
                                           net_table),
            }
            if reference:
                components.append(component)
        return components

    @staticmethod
    def _extract_reference(fp: SExpr) -> str:
        # KiCad 8+: (property "Reference" "R1" ...)
        for prop in children(fp, "property"):
            vals = atoms(prop)
            if len(vals) >= 2 and vals[0] == "Reference":
                return vals[1]
        # KiCad <= 7: (fp_text reference "R1" ...)
        for txt in children(fp, "fp_text"):
            vals = atoms(txt)
            if len(vals) >= 2 and vals[0] == "reference":
                return vals[1]
        return ""

    @staticmethod
    def _extract_value(fp: SExpr) -> str:
        for prop in children(fp, "property"):
            vals = atoms(prop)
            if len(vals) >= 2 and vals[0] == "Value":
                return vals[1]
        for txt in children(fp, "fp_text"):
            vals = atoms(txt)
            if len(vals) >= 2 and vals[0] == "value":
                return vals[1]
        return ""

    def _extract_pads(self, fp: SExpr, component_ref: str,
                      fx: float, fy: float, fangle: float, back: bool,
                      net_table: Dict[str, str]) -> List[Dict[str, Any]]:
        pads = []
        seen_ids: Dict[str, int] = {}
        for pad_node in children(fp, "pad"):
            vals = atoms(pad_node)
            pad_number = vals[0] if len(vals) > 0 else ""
            pad_type = vals[1] if len(vals) > 1 else "smd"
            pad_shape = vals[2] if len(vals) > 2 else "circle"

            at = child(pad_node, "at")
            at_vals = atoms(at) if at is not None else []
            px = _f(at_vals[0] if len(at_vals) > 0 else None)
            py = _f(at_vals[1] if len(at_vals) > 1 else None)

            wx, wy = self._pad_world(fx, fy, fangle, px, py, back)

            size = child(pad_node, "size")
            size_vals = atoms(size) if size is not None else []
            width = _f(size_vals[0] if len(size_vals) > 0 else None, 1.0)
            height = _f(size_vals[1] if len(size_vals) > 1 else None, width)

            drill = child(pad_node, "drill")
            drill_size = None
            if drill is not None:
                # (drill 0.8) or (drill oval 0.8 1.2)
                dvals = [v for v in atoms(drill) if v != "oval"]
                if dvals:
                    drill_size = _f(dvals[0])

            pad_layer = self._pad_copper_layer(pad_node, back)
            net_name = self._pad_net_name(pad_node, net_table)

            # Uniquify duplicate pad numbers within one footprint (thermal
            # pad splits, NPTH with empty numbers) so pad keys don't collide.
            base_id = f"{component_ref}_{pad_number}"
            count = seen_ids.get(base_id, 0)
            seen_ids[base_id] = count + 1
            pad_id = base_id if count == 0 else f"{base_id}#{count}"

            pads.append({
                "id": pad_id,
                "number": pad_number,
                "type": pad_type,
                "shape": pad_shape,
                "x": wx, "y": wy,
                "width": width, "height": height,
                "drill_size": drill_size,
                "layer": pad_layer,
                "net_id": net_name,  # normalized: net name IS the id
            })
        return pads

    @staticmethod
    def _pad_world(fx: float, fy: float, fangle: float,
                   px: float, py: float, back: bool) -> "tuple":
        """Compose a pad's footprint-local (at px py) into world coordinates.

        KiCad stores pad offsets in the footprint frame, ALREADY in the
        flipped frame for back-side footprints (no extra mirror), and the
        footprint angle is CCW-positive in KiCad's Y-down board frame.
        Verified pad-exact (<=1um) against pcbnew 10.0.4 on real KiCad 10
        boards (front/back, rot 0/90/180/270).
        """
        del back  # placement is identical for both sides
        if fangle:
            a = math.radians(fangle)
            c, s = math.cos(a), math.sin(a)
            rx = px * c + py * s
            ry = -px * s + py * c
        else:
            rx, ry = px, py
        return fx + rx, fy + ry

    @staticmethod
    def _pad_copper_layer(pad_node: SExpr, back: bool) -> str:
        """First copper layer a pad sits on ('F.Cu'/'B.Cu'/'*.Cu' -> F.Cu)."""
        layers_node = child(pad_node, "layers")
        if layers_node is not None:
            for name in atoms(layers_node):
                if name.endswith(".Cu"):
                    if name.startswith("*"):
                        return "F.Cu"  # through-hole: spans all copper
                    return name
        return "B.Cu" if back else "F.Cu"

    @staticmethod
    def _collect_nets(net_table: Dict[str, str],
                      components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Union of the legacy net table and names referenced by pads.

        KiCad 10 has no net table, so nets must be synthesized from pad
        references. Sorted by name for deterministic downstream ordering.
        """
        names = {name for name in net_table.values() if name}
        for comp in components:
            for pad in comp["pads"]:
                if pad["net_id"]:
                    names.add(pad["net_id"])
        return [{"id": name, "name": name, "netclass": "Default"}
                for name in sorted(names)]

    @staticmethod
    def _extract_design_rules(root: SExpr, path: Optional[Path] = None) -> Dict[str, Any]:
        rules = {
            "min_track_width": 0.1,
            "min_track_spacing": 0.1,
            "min_via_diameter": 0.2,
            "min_via_drill": 0.1,
            "min_via_annular_width": 0.05,
            "default_track_width": 0.2,
            "default_clearance": 0.2,
            "default_via_diameter": 0.8,
            "default_via_drill": 0.4,
            "netclasses": {},
        }
        # Legacy dialects kept rules in (setup ...).
        setup = child(root, "setup")
        if setup is not None:
            for key in ("min_track_width", "min_via_diameter", "min_via_drill",
                        "default_track_width", "default_via_diameter",
                        "default_via_drill"):
                val = first_atom(child(setup, key))
                if val is not None:
                    rules[key] = _f(val, rules[key])
        # KiCad 6+ keeps the authoritative rules in the sibling .kicad_pro.
        if path is not None:
            pro = Path(path).with_suffix(".kicad_pro")
            if pro.exists():
                try:
                    import json
                    settings = json.loads(pro.read_text(encoding="utf-8"))
                    pro_rules = (settings.get("board", {})
                                 .get("design_settings", {}).get("rules", {}))
                    for src_key, dst_key in (
                            ("min_track_width", "min_track_width"),
                            ("min_via_diameter", "min_via_diameter"),
                            ("min_through_hole_diameter", "min_via_drill"),
                            ("min_via_annular_width", "min_via_annular_width"),
                            ("min_clearance", "min_track_spacing")):
                        if src_key in pro_rules:
                            rules[dst_key] = float(pro_rules[src_key])
                    logger.info(f"Merged design rules from {pro.name}")
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not read {pro.name}: {e}")
        return rules

    def _extract_tracks(self, root: SExpr,
                        net_table: Dict[str, str]) -> List[Dict[str, Any]]:
        tracks = []
        for seg in children(root, "segment"):
            start = atoms(child(seg, "start") or [])
            end = atoms(child(seg, "end") or [])
            tracks.append({
                "start": {"x": _f(start[0] if len(start) > 0 else None),
                          "y": _f(start[1] if len(start) > 1 else None)},
                "end": {"x": _f(end[0] if len(end) > 0 else None),
                        "y": _f(end[1] if len(end) > 1 else None)},
                "width": _f(first_atom(child(seg, "width")), 0.25),
                "layer": first_atom(child(seg, "layer")) or "F.Cu",
                "net": self._pad_net_name(seg, net_table) or "",
            })
        return tracks

    def _extract_vias(self, root: SExpr,
                      net_table: Dict[str, str]) -> List[Dict[str, Any]]:
        vias = []
        for via in children(root, "via"):
            at = atoms(child(via, "at") or [])
            layer_names = atoms(child(via, "layers") or [])
            vias.append({
                "x": _f(at[0] if len(at) > 0 else None),
                "y": _f(at[1] if len(at) > 1 else None),
                "size": _f(first_atom(child(via, "size")), 0.6),
                "drill": _f(first_atom(child(via, "drill")), 0.3),
                "layers": layer_names or ["F.Cu", "B.Cu"],
                "net": self._pad_net_name(via, net_table) or "",
            })
        return vias

    # ------------------------------------------------------------------
    # Domain conversion
    # ------------------------------------------------------------------

    def create_board_from_data(self, board_data: Dict[str, Any]) -> Board:
        """Create Board domain object from parsed data."""
        copper = [l for l in board_data.get("layers", [])
                  if str(l.get("name", "")).endswith(".Cu")]
        board = Board(
            id="parsed_board",
            name=board_data.get("title", "Parsed Board"),
            thickness=1.6,
            layer_count=len(copper),
        )
        # Stash the merged rules for rule-aware emission (see
        # PathFinderRouter.apply_board_rules); private attr, not a
        # dataclass field.
        board._design_rules = board_data.get("design_rules", {})

        for layer_data in board_data.get("layers", []):
            board.add_layer(Layer(
                name=layer_data["name"],
                type="signal" if str(layer_data["name"]).endswith(".Cu") else "other",
                stackup_position=layer_data.get("stackup_position", 0),
            ))

        for comp_data in board_data.get("components", []):
            component = Component(
                id=comp_data.get("reference", ""),
                reference=comp_data.get("reference", ""),
                value=comp_data.get("value", ""),
                footprint=comp_data.get("footprint", ""),
                position=Coordinate(comp_data.get("x", 0), comp_data.get("y", 0)),
                angle=comp_data.get("angle", 0),
                layer=comp_data.get("layer", "F.Cu"),
            )
            for pad_data in comp_data.get("pads", []):
                component.pads.append(Pad(
                    id=pad_data["id"],
                    component_id=component.id,
                    net_id=pad_data.get("net_id"),
                    position=Coordinate(pad_data.get("x", 0), pad_data.get("y", 0)),
                    size=(pad_data.get("width", 1.0), pad_data.get("height", 1.0)),
                    drill_size=pad_data.get("drill_size"),
                    layer=pad_data.get("layer", "F.Cu"),
                    shape=pad_data.get("shape", "circle"),
                ))
            board.add_component(component)

        for net_data in board_data.get("nets", []):
            net_pads = [pad
                        for component in board.components
                        for pad in component.pads
                        if pad.net_id == net_data["id"]]
            if net_pads:
                board.add_net(Net(
                    id=net_data["id"],
                    name=net_data["name"],
                    netclass=net_data.get("netclass", "Default"),
                    pads=net_pads,
                ))

        return board

    def _convert_to_domain_board(self, board_data: Dict[str, Any],
                                 file_path: str) -> Board:
        """Convert parsed board data to domain Board object (legacy alias)."""
        return self.create_board_from_data(board_data)
