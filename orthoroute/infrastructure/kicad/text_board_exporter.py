"""Deterministic KiCad board export without requiring pcbnew Python bindings."""

import hashlib
import csv
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
    consolidate_mechanical_vias: bool,
) -> Tuple[str, int, int]:
    tracks = geometry["tracks"]
    vias = geometry["vias"]
    if consolidate_mechanical_vias:
        vias = _consolidate_mechanical_vias(vias, copper_layers)
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


def _consolidate_mechanical_vias(
    vias: Sequence[Dict[str, Any]],
    copper_layers: Sequence[str],
) -> List[Dict[str, Any]]:
    """Merge touching same-position CNC spans into one deeper drilled via."""
    layer_index = {
        name: index for index, name in enumerate(copper_layers)
    }
    untouched: List[Tuple[int, Dict[str, Any]]] = []
    groups: Dict[
        Tuple[str, float, float],
        List[Tuple[int, int, int, Dict[str, Any]]],
    ] = {}
    for original_index, item in enumerate(vias):
        if item.get("via_process") != "mechanical_blind_buried":
            untouched.append((original_index, dict(item)))
            continue
        low = min(
            layer_index[item["from_layer"]],
            layer_index[item["to_layer"]],
        )
        high = max(
            layer_index[item["from_layer"]],
            layer_index[item["to_layer"]],
        )
        key = (
            str(item["net"]),
            float(item["x"]),
            float(item["y"]),
        )
        groups.setdefault(key, []).append(
            (low, high, original_index, item)
        )

    consolidated: List[Tuple[int, Dict[str, Any]]] = list(untouched)
    for entries in groups.values():
        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        run = [entries[0]]
        run_low, run_high = entries[0][0], entries[0][1]
        for entry in entries[1:]:
            low, high = entry[0], entry[1]
            if low <= run_high:
                run.append(entry)
                run_high = max(run_high, high)
            else:
                consolidated.append(_merged_via_run(
                    run, run_low, run_high, copper_layers
                ))
                run = [entry]
                run_low, run_high = low, high
        consolidated.append(_merged_via_run(
            run, run_low, run_high, copper_layers
        ))
    consolidated.sort(key=lambda entry: entry[0])
    return [item for _, item in consolidated]


def _merged_via_run(
    run: Sequence[Tuple[int, int, int, Dict[str, Any]]],
    low: int,
    high: int,
    copper_layers: Sequence[str],
) -> Tuple[int, Dict[str, Any]]:
    first_index = min(entry[2] for entry in run)
    merged = dict(min(run, key=lambda entry: entry[2])[3])
    merged["from_layer"] = copper_layers[low]
    merged["to_layer"] = copper_layers[high]
    merged["diameter"] = max(float(entry[3]["diameter"]) for entry in run)
    merged["drill"] = max(float(entry[3]["drill"]) for entry in run)
    if len(run) > 1:
        merged["consolidated_vias"] = len(run)
    return first_index, merged


def _via_span_rows(
    vias: Sequence[Dict[str, Any]],
    copper_layers: Sequence[str],
) -> List[Dict[str, Any]]:
    """Summarize the mechanical drill schedule visible to KiCad/DFM."""
    layer_index = {
        name: index for index, name in enumerate(copper_layers)
    }
    grouped: Dict[Tuple[Any, ...], int] = {}
    for item in vias:
        low = min(
            layer_index[item["from_layer"]],
            layer_index[item["to_layer"]],
        )
        high = max(
            layer_index[item["from_layer"]],
            layer_index[item["to_layer"]],
        )
        if low == 0 and high == len(copper_layers) - 1:
            via_type = "through"
        elif low == 0 or high == len(copper_layers) - 1:
            via_type = "blind"
        else:
            via_type = "buried"
        key = (
            low,
            high,
            via_type,
            str(item.get("via_process", "unspecified")),
            str(item.get("via_kind", "")),
            float(item["diameter"]),
            float(item["drill"]),
        )
        grouped[key] = grouped.get(key, 0) + 1

    rows = []
    for key, count in sorted(grouped.items()):
        (
            low,
            high,
            via_type,
            process,
            kind,
            diameter,
            drill,
        ) = key
        rows.append({
            "from_layer": copper_layers[low],
            "to_layer": copper_layers[high],
            "from_index": low,
            "to_index": high,
            "copper_layers_spanned": high - low + 1,
            "dielectric_gaps_spanned": high - low,
            "via_type": via_type,
            "via_process": process,
            "via_kind": kind,
            "diameter_mm": diameter,
            "drill_mm": drill,
            "count": count,
        })
    return rows


def _write_fabrication_manifest(
    output_path: Path,
    geometry: Dict[str, Any],
    copper_layers: Sequence[str],
    thickness_mm: float,
    consolidate_mechanical_vias: bool,
    limit: Optional[int],
) -> Dict[str, str]:
    """Write the exact exported via schedule for PCBWay DFM review."""
    vias = geometry["vias"]
    if consolidate_mechanical_vias:
        vias = _consolidate_mechanical_vias(vias, copper_layers)
    if limit is not None:
        vias = vias[:limit]
    rows = _via_span_rows(vias, copper_layers)
    stem = output_path.with_name(output_path.stem + "-fabrication")
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    markdown_path = stem.with_suffix(".md")
    payload = {
        "board": str(output_path),
        "fabrication_profile": "pcbway_advanced_hdi_mechanical",
        "qualification": "preliminary_requires_pcbway_dfm_approval",
        "layer_count": len(copper_layers),
        "copper_layers": list(copper_layers),
        "board_thickness_mm": float(thickness_mm),
        "grid_pitch_mm": 0.4,
        "mechanical_vias_consolidated": consolidate_mechanical_vias,
        "via_topology": (
            "contiguous adjacent graph hops are consolidated into one "
            "KiCad mechanical blind/buried via"
            if consolidate_mechanical_vias else
            "adjacent graph hops remain separate KiCad via objects"
        ),
        "fabricator_approval_required": True,
        "via_span_schedule": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(rows[0]) if rows else [
            "from_layer",
            "to_layer",
            "count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0]) if rows else [
        "from_layer",
        "to_layer",
        "count",
    ]
    lines = [
        "# Preliminary PCBWay mechanical via-span schedule",
        "",
        f"- Board: `{output_path.name}`",
        f"- Copper layers: {len(copper_layers)}",
        f"- Thickness: {float(thickness_mm):g} mm",
        "- Grid pitch: 0.4 mm",
        "- Status: requires PCBWay stackup/DFM approval",
        "- Interpretation: contiguous adjacent routing hops are exported as "
        "one normal mechanical blind/buried via.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    lines.append("")
    markdown_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "fabrication_manifest_json": str(json_path),
        "fabrication_manifest_csv": str(csv_path),
        "fabrication_manifest_markdown": str(markdown_path),
    }


def export_geometry_to_board(
    geometry_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    layer_count: int,
    thickness_mm: float = 1.6,
    limit: Optional[int] = None,
    consolidate_mechanical_vias: bool = True,
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
        geometry,
        source_sha256,
        copper_layers,
        limit,
        consolidate_mechanical_vias,
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
    fabrication_manifests = _write_fabrication_manifest(
        output_path,
        geometry,
        copper_layers,
        thickness_mm,
        consolidate_mechanical_vias,
        limit,
    )
    return {
        "output": str(output_path),
        "layers": layer_count,
        "thickness_mm": thickness_mm,
        "tracks": track_count,
        "vias": via_count,
        "mechanical_vias_consolidated": consolidate_mechanical_vias,
        "bytes": output_path.stat().st_size,
        "source_sha256": source_sha256,
        **fabrication_manifests,
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
