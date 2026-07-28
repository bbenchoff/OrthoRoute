"""Build CSV, Markdown, and SVG comparisons from monster progress journals."""

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
)


def _is_candidate_name(name: str) -> bool:
    """Include reduced-layer gates and every full-board layer candidate."""
    if "HDI-pcbway_" not in name:
        return False
    return bool(
        re.search(r"Backplane-(?:8L-1024N|14L-1024N|(\d+)L-\1L)", name)
    )


def _label(path: Path, journal: Dict[str, Any]) -> str:
    name = journal.get("run_name", path.stem)
    layers = re.search(r"Backplane-(\d+)L", name)
    nets = re.search(r"-(\d+)N-", name)
    stamp = re.search(r"-(\d{8})_(\d{6})$", name)
    process = (
        "mechanical" if "pcbway_mechanical" in name else
        "ELIC" if "pcbway_elic" in name else "unspecified"
    )
    scope = f"{nets.group(1)} nets" if nets else "8,192 nets"
    identity = str(journal.get("git_sha", ""))[:7]
    if stamp:
        clock = stamp.group(2)
        identity = " ".join(filter(None, (
            identity,
            f"{clock[:2]}:{clock[2:4]}",
        )))
    suffix = f" {identity}" if identity else ""
    return (
        f"{layers.group(1) if layers else '?'}L {scope} "
        f"{process}{suffix}"
    )


def _load(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(paths, key=lambda item: item.stat().st_mtime):
        journal = json.loads(path.read_text(encoding="utf-8"))
        iterations = journal.get("iterations", [])
        if not iterations and journal.get("status") != "starting":
            continue
        runs.append({
            "path": path,
            "journal": journal,
            "iterations": iterations,
            "label": _label(path, journal),
        })
    return runs


def _row(run: Dict[str, Any]) -> Dict[str, Any]:
    journal = run["journal"]
    iterations = run["iterations"]
    final = iterations[-1] if iterations else {}
    overuse = [int(item.get("overuse_total", 0)) for item in iterations]
    barrels = [
        int(item.get("barrel_conflicts", 0)) for item in iterations
    ]
    path_nodes = [
        int(item.get("path_node_conflicts", 0))
        for item in iterations
    ]
    best_overuse = (
        min(iterations, key=lambda item: int(item.get("overuse_total", 0)))
        if iterations else {}
    )
    best_physical = (
        min(iterations, key=lambda item: int(item.get("barrel_conflicts", 0)))
        if iterations else {}
    )
    best_path_nodes = (
        min(
            iterations,
            key=lambda item: int(
                item.get("path_node_conflicts", 0)
            ),
        )
        if iterations else {}
    )
    completion = journal.get("completion", {})
    routed = final.get("routed_nets", journal.get("routed_nets", 0))
    scope_match = re.search(r"(\d[\d,]*) nets", run["label"])
    target = int(scope_match.group(1).replace(",", "")) if scope_match else 0
    return {
        "run": run["label"],
        "status": journal.get("status", "unknown"),
        "iterations": len(iterations),
        "routed_nets": routed or 0,
        "target_nets": completion.get("total_nets", target),
        "complete": bool(completion.get("complete", False)),
        "initial_overuse": overuse[0] if overuse else "",
        "best_overuse": min(overuse) if overuse else "",
        "best_overuse_iteration": best_overuse.get("iteration", ""),
        "final_overuse": overuse[-1] if overuse else "",
        "initial_physical": barrels[0] if barrels else "",
        "best_physical": min(barrels) if barrels else "",
        "best_physical_iteration": best_physical.get("iteration", ""),
        "final_physical": barrels[-1] if barrels else "",
        "initial_path_nodes": path_nodes[0] if path_nodes else "",
        "best_path_nodes": min(path_nodes) if path_nodes else "",
        "best_path_nodes_iteration": best_path_nodes.get(
            "iteration", ""
        ),
        "final_path_nodes": path_nodes[-1] if path_nodes else "",
        "final_pres_fac": final.get("pres_fac", ""),
        "elapsed_seconds": final.get(
            "elapsed_seconds", journal.get("elapsed_seconds", "")
        ),
        "progress_file": str(run["path"]),
    }


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = (
        "run", "status", "iterations", "routed_nets", "target_nets",
        "complete", "initial_overuse", "best_overuse", "final_overuse",
        "best_overuse_iteration", "initial_physical", "best_physical",
        "best_physical_iteration", "final_physical",
        "initial_path_nodes", "best_path_nodes",
        "best_path_nodes_iteration", "final_path_nodes",
        "final_pres_fac",
        "elapsed_seconds",
    )
    lines = [
        "# Reduced-layer monster route comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    lines.extend([
        "",
        "Generated from the per-iteration progress JSON journals. "
        "A `routing` status can mean either an active process or a preserved "
        "stopped checkpoint; consult `CODEX_SUMMARY.md` for lifecycle notes.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _points(
    values: Sequence[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    max_iterations: int,
    max_value: float,
    *,
    log_scale: bool = True,
) -> str:
    result = []
    log_max = math.log10(max(1.0, max_value))
    for index, value in enumerate(values, start=1):
        x = x0 + width * (index - 1) / max(1, max_iterations - 1)
        if log_scale:
            normalized = (
                math.log10(max(1.0, value)) / max(1.0, log_max)
            )
        else:
            normalized = max(0.0, value) / max(1.0, max_value)
        y = y0 + height * (1.0 - normalized)
        result.append(f"{x:.1f},{y:.1f}")
    return " ".join(result)


def _write_svg(runs: Sequence[Dict[str, Any]], path: Path) -> None:
    width, height = 1200, 1340
    plot_x, plot_width = 90, 1040
    panel_height = 200
    panels = [
        ("Graph overuse (log scale)", "overuse_total", 90, True),
        (
            "Physical conflict reports (log scale)",
            "barrel_conflicts",
            390,
            True,
        ),
        (
            "Shared capacity-one path nodes (log scale)",
            "path_node_conflicts",
            690,
            True,
        ),
        ("Present congestion pressure", "pres_fac", 990, False),
    ]
    max_iterations = max(
        (len(run["iterations"]) for run in runs), default=1
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="38" font-family="sans-serif" font-size="24" '
        'font-weight="bold">Reduced-layer monster routing progress</text>',
    ]
    for title, key, y0, log_scale in panels:
        maximum = max(
            (
                float(item.get(key, 0))
                for run in runs
                for item in run["iterations"]
            ),
            default=1.0,
        )
        svg.extend([
            f'<text x="{plot_x}" y="{y0 - 18}" font-family="sans-serif" '
            f'font-size="16">{html.escape(title)}</text>',
            f'<rect x="{plot_x}" y="{y0}" width="{plot_width}" '
            f'height="{panel_height}" fill="#fafafa" stroke="#cbd5e1"/>',
        ])
        for tick in range(5):
            y = y0 + panel_height * tick / 4
            if log_scale:
                exponent = math.log10(maximum) * (1.0 - tick / 4)
                value = f"{int(10 ** exponent):,}"
            else:
                value = f"{maximum * (1.0 - tick / 4):.1f}"
            svg.extend([
                f'<line x1="{plot_x}" y1="{y:.1f}" '
                f'x2="{plot_x + plot_width}" y2="{y:.1f}" '
                'stroke="#e2e8f0"/>',
                f'<text x="{plot_x - 8}" y="{y + 4:.1f}" '
                'text-anchor="end" font-family="monospace" font-size="11">'
                f'{value}</text>',
            ])
        for run_index, run in enumerate(runs):
            values = [
                float(item.get(key, 0)) for item in run["iterations"]
            ]
            if not values:
                continue
            points = _points(
                values, plot_x, y0, plot_width, panel_height,
                max_iterations, maximum, log_scale=log_scale,
            )
            color = COLORS[run_index % len(COLORS)]
            svg.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        for tick in range(0, max_iterations + 1, 10):
            x = plot_x + plot_width * tick / max(1, max_iterations)
            svg.append(
                f'<text x="{x:.1f}" y="{y0 + panel_height + 18}" '
                'text-anchor="middle" font-family="sans-serif" '
                f'font-size="11">{tick}</text>'
            )
    legend_y = 1260
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        x = 80 + (index % 3) * 370
        y = legend_y + (index // 3) * 24
        svg.extend([
            f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" '
            f'stroke="{color}" stroke-width="4"/>',
            f'<text x="{x + 32}" y="{y + 4}" font-family="sans-serif" '
            f'font-size="12">{html.escape(run["label"])}</text>',
        ])
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    candidates = [
        path for path in args.results_dir.glob("*progress.json")
        if _is_candidate_name(path.name)
    ]
    runs = _load(candidates)
    if not runs:
        raise SystemExit("no matching progress journals found")
    rows = [_row(run) for run in runs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "reduced-layer-comparison.csv")
    _write_markdown(rows, args.output_dir / "reduced-layer-comparison.md")
    _write_svg(runs, args.output_dir / "reduced-layer-progress.svg")
    print(json.dumps({
        "runs": len(runs),
        "csv": str(args.output_dir / "reduced-layer-comparison.csv"),
        "markdown": str(args.output_dir / "reduced-layer-comparison.md"),
        "svg": str(args.output_dir / "reduced-layer-progress.svg"),
    }, indent=2))


if __name__ == "__main__":
    main()
