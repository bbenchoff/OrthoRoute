"""Build the accepted-board layer/drill/wirelength cost curve.

The monster sweep state is the authority tying a qualified KiCad board to its
route journal, metrics, DRC result, and fabrication manifest.  This module
reduces those accepted rungs to stable CSV, Markdown, and SVG artifacts.
"""

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    seen_layers = set()
    accepted_runs = [
        run for run in state.get("runs", []) if run.get("accepted")
    ]
    for run in sorted(accepted_runs, key=lambda item: int(item["layers"])):
        layer_count = int(run["layers"])
        if layer_count in seen_layers:
            continue
        qualification = run.get("qualification", {})
        deliverable = qualification.get("deliverable", {})
        progress = Path(run["progress"])
        journal = _read_json(progress)
        metrics_path = journal.get("metrics")
        if not metrics_path:
            raise ValueError(f"{progress} has no metrics artifact")
        metrics = _read_json(Path(metrics_path))
        manifest_path = Path(deliverable["fabrication_manifest_json"])
        manifest = _read_json(manifest_path)
        schedule = manifest.get("via_span_schedule", [])
        span_histogram = Counter()
        span_classes = Counter()
        drill_count = 0
        for span in schedule:
            count = int(span.get("count", 0))
            drill_count += count
            gaps = int(span.get("dielectric_gaps_spanned", 0))
            via_type = str(span.get("via_type", "unknown"))
            span_histogram[f"{via_type}/{gaps}-gap"] += count
            span_classes[
                f"{span.get('from_layer')}->{span.get('to_layer')}"
            ] += count
        warm_start = journal.get("warm_start") or {}
        rows.append({
            "total_layers": layer_count,
            "route_kind": run.get("reason", "unknown"),
            "route_passes": int(journal.get("iteration", 0)),
            "wall_time_s": round(float(journal.get("elapsed_seconds", 0)), 3),
            "displaced_net_count": int(
                warm_start.get("displaced_net_count", 0)
            ),
            "displaced_net_fraction": round(
                float(warm_start.get("displaced_net_count", 0)) / 8192,
                6,
            ),
            "wirelength_mm": float(metrics["copper"]["wirelength_mm"]),
            "via_drill_count": drill_count,
            "via_layer_steps": int(metrics["copper"]["via_layer_steps"]),
            "drc_warning_count": int(deliverable["drc_warnings"]),
            "drc_reported_errors": int(deliverable["reported_errors"]),
            "span_type_histogram": json.dumps(
                dict(sorted(span_histogram.items())),
                separators=(",", ":"),
            ),
            "lamination_span_classes": json.dumps(
                dict(sorted(span_classes.items())),
                separators=(",", ":"),
            ),
            "board": str(deliverable["board"]),
            "fabrication_manifest": str(manifest_path),
            "drc_report": str(deliverable["drc"]),
            "progress": str(progress),
        })
        seen_layers.add(layer_count)
    return rows


def _write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "total_layers",
        "route_kind",
        "route_passes",
        "wall_time_s",
        "displaced_net_count",
        "displaced_net_fraction",
        "wirelength_mm",
        "via_drill_count",
        "via_layer_steps",
        "drc_warning_count",
        "drc_reported_errors",
        "span_type_histogram",
        "lamination_span_classes",
        "board",
        "fabrication_manifest",
        "drc_report",
        "progress",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        "# Accepted-board layer cost curve",
        "",
        "Only exported KiCad boards below the configured DRC error gate are "
        "included.",
        "",
        "| layers | passes | displaced | wirelength (mm) | drills | "
        "via layer-steps | DRC warnings |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['total_layers']} | {row['route_passes']} | "
        f"{row['displaced_net_count']} "
        f"({100 * row['displaced_net_fraction']:.2f}%) | "
        f"{row['wirelength_mm']:.1f} | {row['via_drill_count']} | "
        f"{row['via_layer_steps']} | {row['drc_warning_count']} |"
        for row in rows
    )
    for row in rows:
        lines.extend([
            "",
            f"## {row['total_layers']} layers",
            "",
            f"- Board: `{row['board']}`",
            f"- Fabrication manifest: `{row['fabrication_manifest']}`",
            f"- Span-type histogram: `{row['span_type_histogram']}`",
            f"- Lamination span classes: "
            f"`{row['lamination_span_classes']}`",
        ])
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _polyline(
    rows: List[Dict[str, Any]],
    key: str,
    *,
    x_for,
    y_min: float,
    y_max: float,
    top: float,
    height: float,
) -> str:
    span = max(y_max - y_min, 1.0)
    points = []
    for row in rows:
        y = top + height - (
            (float(row[key]) - y_min) / span
        ) * height
        points.append(f"{x_for(row['total_layers']):.2f},{y:.2f}")
    return " ".join(points)


def _scaled_y(
    value: float,
    minimum: float,
    maximum: float,
    top: float,
    height: float,
) -> float:
    span = max(maximum - minimum, 1.0)
    return top + height - ((value - minimum) / span) * height


def _write_svg(rows: List[Dict[str, Any]], path: Path) -> None:
    width, height = 1200, 820
    left, right, top, bottom = 130, 110, 100, 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    if rows:
        layers = [row["total_layers"] for row in rows]
        x_min, x_max = min(layers), max(layers)
    else:
        x_min, x_max = 0, 1
    x_span = max(x_max - x_min, 1)

    def x_for(layer):
        if x_max == x_min:
            return left + plot_width / 2
        return left + (layer - x_min) / x_span * plot_width

    wire_values = [row["wirelength_mm"] for row in rows] or [0, 1]
    drill_values = [row["via_drill_count"] for row in rows] or [0, 1]
    wire_min, wire_max = min(wire_values), max(wire_values)
    drill_min, drill_max = min(drill_values), max(drill_values)
    wire_points = _polyline(
        rows,
        "wirelength_mm",
        x_for=x_for,
        y_min=wire_min,
        y_max=wire_max,
        top=top,
        height=plot_height,
    )
    drill_points = _polyline(
        rows,
        "via_drill_count",
        x_for=x_for,
        y_min=drill_min,
        y_max=drill_max,
        top=top,
        height=plot_height,
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="60" y="52" font-family="sans-serif" font-size="30" '
        'font-weight="700" fill="#0f172a">Accepted-board layer cost '
        'curve</text>',
        f'<line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" '
        'stroke="#475569" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" stroke="#475569" stroke-width="2"/>',
        f'<polyline points="{wire_points}" fill="none" stroke="#2563eb" '
        'stroke-width="4"/>',
        f'<polyline points="{drill_points}" fill="none" stroke="#dc2626" '
        'stroke-width="4"/>',
    ]
    for row in rows:
        x = x_for(row["total_layers"])
        wire_y = _scaled_y(
            row["wirelength_mm"],
            wire_min,
            wire_max,
            top,
            plot_height,
        )
        drill_y = _scaled_y(
            row["via_drill_count"],
            drill_min,
            drill_max,
            top,
            plot_height,
        )
        elements.extend([
            f'<circle cx="{x:.2f}" cy="{wire_y:.2f}" r="7" '
            'fill="#2563eb"/>',
            f'<circle cx="{x:.2f}" cy="{drill_y:.2f}" r="7" '
            'fill="#dc2626"/>',
            f'<text x="{x:.2f}" y="{top + plot_height + 38}" '
            'font-family="sans-serif" font-size="20" text-anchor="middle" '
            f'fill="#334155">{row["total_layers"]}L</text>',
        ])
    elements.extend([
        f'<text x="{left + plot_width / 2}" y="{height - 45}" '
        'font-family="sans-serif" font-size="22" text-anchor="middle" '
        'fill="#334155">Total copper layers</text>',
        f'<text x="{left}" y="{height - 95}" font-family="sans-serif" '
        'font-size="18" fill="#2563eb">Wirelength: '
        f'{html.escape(f"{wire_min:.1f}-{wire_max:.1f} mm")}</text>',
        f'<text x="{left + 420}" y="{height - 95}" '
        'font-family="sans-serif" font-size="18" fill="#dc2626">'
        f'Drills: {drill_min}-{drill_max}</text>',
        '</svg>',
    ])
    path.write_text(
        "\n".join(elements) + "\n", encoding="utf-8", newline="\n"
    )


def write_layer_cost_curve(
    state: Dict[str, Any],
    output_stem: Path,
) -> Dict[str, Any]:
    rows = _accepted_rows(state)
    csv_path = output_stem.with_suffix(".csv")
    markdown_path = output_stem.with_suffix(".md")
    svg_path = output_stem.with_suffix(".svg")
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path)
    _write_svg(rows, svg_path)
    return {
        "accepted_board_count": len(rows),
        "layers": [row["total_layers"] for row in rows],
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "svg": str(svg_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    result = write_layer_cost_curve(
        _read_json(args.state),
        args.output_stem,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
