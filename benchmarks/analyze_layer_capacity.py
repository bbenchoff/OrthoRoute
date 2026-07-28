#!/usr/bin/env python3
"""Prove necessary strict-layer capacity with endpoint cut bounds.

Every two-terminal net whose endpoints lie on opposite sides of a board cut
must consume at least one planar edge across that cut. This lower bound is
independent of routing order, detours, costs, or PathFinder convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orthoroute.algorithms.manhattan.board_analyzer import (  # noqa: E402
    assign_layer_axes_by_demand,
)
from orthoroute.algorithms.manhattan.pathfinder.kicad_geometry import (  # noqa: E402
    KiCadGeometry,
)
from orthoroute.infrastructure.kicad.file_parser import (  # noqa: E402
    KiCadFileParser,
)


def crossing_profile(
    intervals: Iterable[tuple[int, int]],
    cut_count: int,
) -> np.ndarray:
    """Count endpoint intervals crossing every cut in O(nets + cuts)."""
    delta = np.zeros(cut_count + 1, dtype=np.int64)
    for first, second in intervals:
        low, high = sorted((int(first), int(second)))
        low = max(0, min(cut_count, low))
        high = max(0, min(cut_count, high))
        if high <= low:
            continue
        delta[low] += 1
        delta[high] -= 1
    return np.cumsum(delta[:-1])


def _svg_polyline(
    values: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    maximum: float,
) -> str:
    if not len(values):
        return ""
    denominator = max(1, len(values) - 1)
    points = []
    for index, value in enumerate(values):
        px = x + width * index / denominator
        py = y + height * (1.0 - float(value) / maximum)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def write_svg(
    path: Path,
    h_crossings: np.ndarray,
    v_crossings: np.ndarray,
    rows: list[dict],
) -> None:
    width, height = 1200, 820
    left, plot_width, plot_height = 90, 1030, 270
    panels = [
        (
            85,
            "Vertical cuts: nets requiring horizontal edges",
            h_crossings,
            "h_layers",
            "y_steps",
        ),
        (
            470,
            "Horizontal cuts: nets requiring vertical edges",
            v_crossings,
            "v_layers",
            "x_steps",
        ),
    ]
    colors = ["#dc2626", "#2563eb", "#16a34a", "#9333ea"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Segoe UI,Arial,sans-serif}"
        ".title{font-size:22px;font-weight:700}.label{font-size:14px}"
        ".small{font-size:12px}.grid{stroke:#d1d5db;stroke-width:1}"
        ".axis{stroke:#111827;stroke-width:1.5}</style>",
        '<rect width="1200" height="820" fill="#ffffff"/>',
        '<text class="title" x="600" y="34" text-anchor="middle">'
        "OrthoRoute strict-layer cut-capacity proof (0.4 mm grid)</text>",
    ]
    for panel_index, (
        top,
        title,
        crossings,
        layer_key,
        span_key,
    ) in enumerate(panels):
        capacities = [
            int(row[layer_key]) * int(row[span_key])
            for row in rows
        ]
        maximum = max(
            float(crossings.max(initial=0)),
            max(capacities, default=1),
        ) * 1.08
        elements.append(
            f'<text class="label" x="{left}" y="{top - 18}">{title}</text>'
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            gy = top + plot_height * (1.0 - fraction)
            value = maximum * fraction
            elements.extend([
                f'<line class="grid" x1="{left}" y1="{gy:.2f}" '
                f'x2="{left + plot_width}" y2="{gy:.2f}"/>',
                f'<text class="small" x="{left - 10}" y="{gy + 4:.2f}" '
                f'text-anchor="end">{value:.0f}</text>',
            ])
        elements.extend([
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<polyline fill="none" stroke="#111827" stroke-width="3" '
            f'points="{_svg_polyline(crossings, left, top, plot_width, plot_height, maximum)}"/>',
        ])
        legend_y = top + plot_height + 27
        elements.append(
            f'<line x1="{left}" y1="{legend_y - 5}" x2="{left + 28}" '
            f'y2="{legend_y - 5}" stroke="#111827" stroke-width="3"/>'
        )
        elements.append(
            f'<text class="small" x="{left + 36}" y="{legend_y}">'
            f"required crossings (max {int(crossings.max(initial=0))})</text>"
        )
        legend_x = left + 270
        seen = set()
        for row_index, (row, capacity) in enumerate(zip(rows, capacities)):
            if capacity in seen:
                continue
            seen.add(capacity)
            color = colors[row_index % len(colors)]
            cy = top + plot_height * (1.0 - capacity / maximum)
            elements.append(
                f'<line x1="{left}" y1="{cy:.2f}" '
                f'x2="{left + plot_width}" y2="{cy:.2f}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="8 5"/>'
            )
            label = (
                f'{row["total_layers"]}L strict: '
                f'{row[layer_key]} layers = {capacity} slots'
            )
            elements.append(
                f'<text class="small" x="{legend_x}" y="{legend_y}">'
                f'<tspan fill="{color}">—</tspan> {label}</text>'
            )
            legend_x += 245
    elements.extend([
        '<text class="small" x="600" y="805" text-anchor="middle">'
        "A required-crossing curve above a strict capacity line is an "
        "impossibility proof, not a routing heuristic.</text>",
        "</svg>",
    ])
    path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pitch", type=float, default=0.4)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[16, 18, 20],
    )
    args = parser.parse_args()

    board = KiCadFileParser().load_board(str(args.board))
    if board is None:
        raise RuntimeError(f"Could not parse {args.board}")
    nets = [
        net for net in board.nets
        if len(getattr(net, "pads", ())) >= 2
    ]
    pads = [pad for net in nets for pad in net.pads]
    margin = 3.0
    bounds = (
        min(pad.position.x for pad in pads) - margin,
        min(pad.position.y for pad in pads) - margin,
        max(pad.position.x for pad in pads) + margin,
        max(pad.position.y for pad in pads) + margin,
    )
    geometry = KiCadGeometry(bounds, args.pitch, max(args.layers))

    x_intervals = []
    y_intervals = []
    h_steps = 0
    v_steps = 0
    for net in nets:
        first, second = net.pads[:2]
        x1, y1 = geometry.world_to_lattice(
            first.position.x, first.position.y
        )
        x2, y2 = geometry.world_to_lattice(
            second.position.x, second.position.y
        )
        x_intervals.append((x1, x2))
        y_intervals.append((y1, y2))
        h_steps += abs(x2 - x1)
        v_steps += abs(y2 - y1)

    h_crossings = crossing_profile(
        x_intervals, geometry.x_steps - 1
    )
    v_crossings = crossing_profile(
        y_intervals, geometry.y_steps - 1
    )
    h_required_cut = math.ceil(
        int(h_crossings.max()) / geometry.y_steps
    )
    v_required_cut = math.ceil(
        int(v_crossings.max()) / geometry.x_steps
    )
    h_required_global = math.ceil(
        h_steps / ((geometry.x_steps - 1) * geometry.y_steps)
    )
    v_required_global = math.ceil(
        v_steps / (geometry.x_steps * (geometry.y_steps - 1))
    )
    h_fraction = h_steps / max(1, h_steps + v_steps)

    rows = []
    for total_layers in sorted(set(args.layers)):
        signal_layers = list(range(1, total_layers - 1))
        h_layers, v_layers = assign_layer_axes_by_demand(
            signal_layers, h_fraction
        )
        h_cut_capacity = len(h_layers) * geometry.y_steps
        v_cut_capacity = len(v_layers) * geometry.x_steps
        h_global_capacity = (
            len(h_layers)
            * (geometry.x_steps - 1)
            * geometry.y_steps
        )
        v_global_capacity = (
            len(v_layers)
            * geometry.x_steps
            * (geometry.y_steps - 1)
        )
        rows.append({
            "total_layers": total_layers,
            "internal_layers": len(signal_layers),
            "h_layers": len(h_layers),
            "v_layers": len(v_layers),
            "x_steps": geometry.x_steps,
            "y_steps": geometry.y_steps,
            "max_h_cut_demand": int(h_crossings.max()),
            "h_cut_capacity": h_cut_capacity,
            "h_cut_utilization": (
                int(h_crossings.max()) / h_cut_capacity
            ),
            "max_v_cut_demand": int(v_crossings.max()),
            "v_cut_capacity": v_cut_capacity,
            "v_cut_utilization": (
                int(v_crossings.max()) / v_cut_capacity
            ),
            "h_global_utilization": h_steps / h_global_capacity,
            "v_global_utilization": v_steps / v_global_capacity,
            "strict_cut_feasible": (
                int(h_crossings.max()) <= h_cut_capacity
                and int(v_crossings.max()) <= v_cut_capacity
            ),
            "guided_h_cut_utilization": (
                int(h_crossings.max())
                / (len(signal_layers) * geometry.y_steps)
            ),
            "guided_v_cut_utilization": (
                int(v_crossings.max())
                / (len(signal_layers) * geometry.x_steps)
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "strict-layer-capacity"
    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    proof = {
        "source_board": str(args.board.resolve()),
        "grid_pitch_mm": args.pitch,
        "routable_nets": len(nets),
        "x_steps": geometry.x_steps,
        "y_steps": geometry.y_steps,
        "horizontal_demand_fraction": h_fraction,
        "raw_horizontal_steps": h_steps,
        "raw_vertical_steps": v_steps,
        "minimum_h_layers_global": h_required_global,
        "minimum_v_layers_global": v_required_global,
        "minimum_h_layers_cut": h_required_cut,
        "minimum_v_layers_cut": v_required_cut,
        "minimum_strict_internal_layers_cut": (
            h_required_cut + v_required_cut
        ),
        "minimum_strict_total_layers_cut": (
            h_required_cut + v_required_cut + 2
        ),
        "max_horizontal_crossing_cut": {
            "index": int(h_crossings.argmax()),
            "demand": int(h_crossings.max()),
        },
        "max_vertical_crossing_cut": {
            "index": int(v_crossings.argmax()),
            "demand": int(v_crossings.max()),
        },
        "candidates": rows,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )

    markdown = [
        "# Strict one-axis-per-layer cut-capacity proof",
        "",
        (
            f"Source: `{args.board}`; {len(nets):,} routable nets; "
            f"{geometry.x_steps} x {geometry.y_steps} nodes at "
            f"{args.pitch:g} mm."
        ),
        "",
        (
            "Necessary cut lower bound: "
            f"{h_required_cut} H + {v_required_cut} V = "
            f"{h_required_cut + v_required_cut} internal layers, "
            f"or {h_required_cut + v_required_cut + 2} total copper "
            "layers."
        ),
        "",
        "| Total | Split | H cut use | V cut use | Strict feasible | "
        "Guided H/V cut use |",
        "|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f'| {row["total_layers"]} | '
            f'{row["h_layers"]}H/{row["v_layers"]}V | '
            f'{row["h_cut_utilization"]:.1%} | '
            f'{row["v_cut_utilization"]:.1%} | '
            f'{"yes" if row["strict_cut_feasible"] else "no"} | '
            f'{row["guided_h_cut_utilization"]:.1%} / '
            f'{row["guided_v_cut_utilization"]:.1%} |'
        )
    markdown.extend([
        "",
        (
            "This is a necessary lower bound only: passing it does not prove "
            "routability, but failing it proves strict routing impossible."
        ),
    ])
    stem.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    write_svg(
        stem.with_suffix(".svg"),
        h_crossings,
        v_crossings,
        rows,
    )
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
