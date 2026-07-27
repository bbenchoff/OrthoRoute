"""Deterministic KiCad board export without requiring pcbnew Python bindings."""

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sexpr import _scan_node_end, find_top_level_spans, strip_top_level_nodes


def _atom(value: Any) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _quoted(value: Any) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _named_node_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """Return all structural nodes named ``name``, ignoring quoted strings."""
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif text[i] == "(":
            j = i + 1
            while j < len(text) and text[j] not in ' \t\r\n()"':
                j += 1
            if text[i + 1:j] == name:
                spans.append((i, _scan_node_end(text, i)))
            i += 1
        else:
            i += 1
    return spans


def _replace_unique_node(text: str, name: str, replacement: str) -> str:
    spans = _named_node_spans(text, name)
    if len(spans) != 1:
        raise ValueError(
            f"expected exactly one ({name} ...) node, found {len(spans)}"
        )
    start, end = spans[0]
    return text[:start] + replacement + text[end:]


def _copper_layers(layer_count: int) -> List[str]:
    if layer_count < 4 or layer_count > 32 or layer_count % 2:
        raise ValueError("KiCad export requires an even 4-32 layer count")
    return (
        ["F.Cu"]
        + [f"In{index}.Cu" for index in range(1, layer_count - 1)]
        + ["B.Cu"]
    )


def _reduce_layer_table(text: str, copper_layers: Sequence[str]) -> str:
    spans = find_top_level_spans(text, ("layers",))
    if len(spans) != 1:
        raise ValueError(
            f"expected exactly one top-level layers node, found {len(spans)}"
        )
    start, end = spans[0]
    keep = set(copper_layers)
    layer_line = re.compile(r'^\s*\(\d+\s+"([^"]+\.Cu)"')
    retained = []
    for line in text[start:end].splitlines(keepends=True):
        match = layer_line.match(line)
        if match is None or match.group(1) in keep:
            retained.append(line)
    return text[:start] + "".join(retained) + text[end:]


def _stackup(copper_layers: Sequence[str], thickness_mm: float) -> str:
    copper_thickness = 0.035
    mask_thickness = 0.01
    core_thickness = 0.19
    gap_count = len(copper_layers) - 1
    prepreg_count = gap_count - 1
    prepreg_thickness = (
        thickness_mm
        - 2.0 * mask_thickness
        - len(copper_layers) * copper_thickness
        - core_thickness
    ) / prepreg_count
    if prepreg_thickness <= 0:
        raise ValueError("board thickness is too small for the copper stack")
    central_gap = len(copper_layers) // 2 - 1

    lines = [
        "(stackup\n",
        '\t\t\t(layer "F.SilkS"\n\t\t\t\t(type "Top Silk Screen")\n\t\t\t)\n',
        '\t\t\t(layer "F.Paste"\n\t\t\t\t(type "Top Solder Paste")\n\t\t\t)\n',
        '\t\t\t(layer "F.Mask"\n\t\t\t\t(type "Top Solder Mask")\n'
        f"\t\t\t\t(thickness {_atom(mask_thickness)})\n\t\t\t)\n",
    ]
    for index, layer in enumerate(copper_layers):
        lines.append(
            f"\t\t\t(layer {_quoted(layer)}\n"
            '\t\t\t\t(type "copper")\n'
            f"\t\t\t\t(thickness {_atom(copper_thickness)})\n"
            "\t\t\t)\n"
        )
        if index == len(copper_layers) - 1:
            continue
        dielectric_type = "core" if index == central_gap else "prepreg"
        dielectric_thickness = (
            core_thickness if index == central_gap else prepreg_thickness
        )
        lines.append(
            f'\t\t\t(layer "dielectric {index + 1}"\n'
            f'\t\t\t\t(type "{dielectric_type}")\n'
            f"\t\t\t\t(thickness {_atom(dielectric_thickness)})\n"
            '\t\t\t\t(material "FR4")\n'
            "\t\t\t\t(epsilon_r 4.5)\n"
            "\t\t\t\t(loss_tangent 0.02)\n"
            "\t\t\t)\n"
        )
    lines.extend([
        '\t\t\t(layer "B.Mask"\n\t\t\t\t(type "Bottom Solder Mask")\n'
        f"\t\t\t\t(thickness {_atom(mask_thickness)})\n\t\t\t)\n",
        '\t\t\t(layer "B.Paste"\n\t\t\t\t(type "Bottom Solder Paste")\n\t\t\t)\n',
        '\t\t\t(layer "B.SilkS"\n\t\t\t\t(type "Bottom Silk Screen")\n\t\t\t)\n',
        '\t\t\t(copper_finish "None")\n',
        "\t\t\t(dielectric_constraints no)\n",
        "\t\t)",
    ])
    return "".join(lines)


def _set_general_thickness(text: str, thickness_mm: float) -> str:
    spans = _named_node_spans(text, "general")
    if len(spans) != 1:
        raise ValueError(
            f"expected exactly one (general ...) node, found {len(spans)}"
        )
    start, end = spans[0]
    node = text[start:end]
    replaced, count = re.subn(
        r"(\(thickness\s+)[^)]+",
        rf"\g<1>{_atom(thickness_mm)}",
        node,
        count=1,
    )
    if count != 1:
        raise ValueError("general node has no board thickness")
    return text[:start] + replaced + text[end:]


def _geometry_nodes(
    geometry: Dict[str, Any],
    source_sha256: str,
    copper_layers: Sequence[str],
    limit: Optional[int],
) -> Tuple[str, int, int]:
    tracks = geometry["tracks"]
    vias = geometry["vias"]
    if limit is not None:
        tracks = tracks[:limit]
        vias = vias[:limit]
    allowed = set(copper_layers)
    nodes = []
    for index, item in enumerate(tracks):
        if item["layer"] not in allowed:
            raise ValueError(f"track uses omitted layer {item['layer']}")
        item_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{source_sha256}:segment:{index}"
        )
        nodes.append(
            "\t(segment\n"
            f"\t\t(start {_atom(item['x1'])} {_atom(item['y1'])})\n"
            f"\t\t(end {_atom(item['x2'])} {_atom(item['y2'])})\n"
            f"\t\t(width {_atom(item['width'])})\n"
            f"\t\t(layer {_quoted(item['layer'])})\n"
            f"\t\t(net {_quoted(item['net'])})\n"
            f"\t\t(uuid {_quoted(item_uuid)})\n"
            "\t)\n"
        )
    for index, item in enumerate(vias):
        from_layer = item["from_layer"]
        to_layer = item["to_layer"]
        if from_layer not in allowed or to_layer not in allowed:
            raise ValueError(
                f"via uses omitted layers {from_layer}->{to_layer}"
            )
        process = item.get("via_process", "")
        kind = item.get("via_kind", "")
        if {from_layer, to_layer} == {"F.Cu", "B.Cu"}:
            via_type = ""
        elif process == "laser_microvia" or kind == "microvia":
            via_type = " micro"
        else:
            via_type = " blind"
        item_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{source_sha256}:via:{index}"
        )
        nodes.append(
            f"\t(via{via_type}\n"
            f"\t\t(at {_atom(item['x'])} {_atom(item['y'])})\n"
            f"\t\t(size {_atom(item['diameter'])})\n"
            f"\t\t(drill {_atom(item['drill'])})\n"
            f"\t\t(layers {_quoted(from_layer)} {_quoted(to_layer)})\n"
            f"\t\t(net {_quoted(item['net'])})\n"
            f"\t\t(uuid {_quoted(item_uuid)})\n"
            "\t)\n"
        )
    return "".join(nodes), len(tracks), len(vias)


def export_geometry_to_board(
    geometry_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    layer_count: int,
    thickness_mm: float = 1.6,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a reduced-layer KiCad board containing routed geometry."""
    if output_path.resolve() == source_path.resolve():
        raise ValueError("refusing to overwrite the source board")
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    expected = geometry.get("source_sha256")
    if expected and expected.lower() != source_sha256.lower():
        raise ValueError(
            f"source hash mismatch: expected {expected}, got {source_sha256}"
        )

    copper_layers = _copper_layers(layer_count)
    text = source_bytes.decode("utf-8")
    text, _ = strip_top_level_nodes(text, ("segment", "via"))
    text = _set_general_thickness(text, thickness_mm)
    text = _reduce_layer_table(text, copper_layers)
    text = _replace_unique_node(
        text, "stackup", _stackup(copper_layers, thickness_mm)
    )
    for index in range(layer_count - 1, 31):
        removed = f"In{index}.Cu"
        if f'"{removed}"' in text:
            raise ValueError(f"source object still references {removed}")

    nodes, track_count, via_count = _geometry_nodes(
        geometry, source_sha256, copper_layers, limit
    )
    root_end = text.rfind("\n)")
    if root_end < 0 or text[root_end + 2:].strip():
        raise ValueError("board does not end with one root close")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        text[:root_end] + "\n" + nodes + text[root_end:],
        encoding="utf-8",
        newline="\n",
    )
    return {
        "output": str(output_path),
        "layers": layer_count,
        "thickness_mm": thickness_mm,
        "tracks": track_count,
        "vias": via_count,
        "bytes": output_path.stat().st_size,
        "source_sha256": source_sha256,
    }


def export_project_for_geometry(
    geometry_path: Path,
    source_project_path: Path,
    output_project_path: Path,
) -> Dict[str, Any]:
    """Copy a KiCad project with constraints matching routed geometry."""
    if output_project_path.resolve() == source_project_path.resolve():
        raise ValueError("refusing to overwrite the source project")
    project = json.loads(source_project_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    tracks = geometry["tracks"]
    vias = geometry["vias"]
    if not tracks or not vias:
        raise ValueError("geometry requires tracks and vias")

    track_width = min(float(item["width"]) for item in tracks)
    annular = min(
        0.5 * (float(item["diameter"]) - float(item["drill"]))
        for item in vias
    )
    mechanical = [
        item for item in vias
        if item.get("via_process") != "laser_microvia"
        and item.get("via_kind") != "microvia"
    ]
    microvias = [
        item for item in vias
        if item.get("via_process") == "laser_microvia"
        or item.get("via_kind") == "microvia"
    ]
    rules = project["board"]["design_settings"]["rules"]
    rules.update({
        "min_clearance": 0.1,
        "min_hole_clearance": 0.1524,
        "min_hole_to_hole": 0.2794,
        "min_track_width": track_width,
        "min_via_annular_width": annular,
        "min_via_diameter": min(float(v["diameter"]) for v in vias),
        "min_through_hole_diameter": min(
            float(v["drill"]) for v in mechanical or vias
        ),
    })
    if microvias:
        rules.update({
            "min_microvia_diameter": min(
                float(v["diameter"]) for v in microvias
            ),
            "min_microvia_drill": min(
                float(v["drill"]) for v in microvias
            ),
        })

    classes = project["net_settings"]["classes"]
    if isinstance(classes, dict):
        classes = [classes]
    default_class = next(
        (item for item in classes if item.get("name") == "Default"),
        classes[0],
    )
    default_class["clearance"] = 0.1
    default_class["track_width"] = track_width
    default_via = mechanical[0] if mechanical else vias[0]
    default_class["via_diameter"] = float(default_via["diameter"])
    default_class["via_drill"] = float(default_via["drill"])
    if microvias:
        default_class["microvia_diameter"] = float(
            microvias[0]["diameter"]
        )
        default_class["microvia_drill"] = float(microvias[0]["drill"])

    output_project_path.parent.mkdir(parents=True, exist_ok=True)
    output_project_path.write_text(
        json.dumps(project, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "project": str(output_project_path),
        "track_width_mm": track_width,
        "clearance_mm": 0.1,
        "min_hole_clearance_mm": 0.1524,
        "min_hole_to_hole_mm": 0.2794,
        "min_annular_width_mm": annular,
    }
