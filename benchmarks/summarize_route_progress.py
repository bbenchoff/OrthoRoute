"""Build CSV, Markdown, and SVG comparisons from monster progress journals."""

import argparse
import csv
import html
import json
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
REPO_ROOT = Path(__file__).resolve().parents[1]
UNIQUE_EDGE_ACCOUNTING_COMMIT = (
    "43e34e0239cab4b7e0ad29e5ed1ac7e8e8baed51"
)
BIDIRECTIONAL_EDGE_RESERVATION_COMMIT = (
    "76eaefd"
)
STALL_WINDOW = 8
STALL_COLUMNS = (
    "layers",
    "run",
    "git_sha",
    "status",
    "edge_accounting",
    "iterations",
    "observation",
    "stall_window",
    "stall_iteration",
    "overuse_at_stall",
    "best_iteration",
    "best_overuse",
    "current_overuse",
    "eligible_for_fit",
    "progress_file",
)


@lru_cache(maxsize=None)
def _uses_unique_edge_accounting(git_sha: str) -> bool:
    if not git_sha:
        return False
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            UNIQUE_EDGE_ACCOUNTING_COMMIT,
            git_sha,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


@lru_cache(maxsize=None)
def _edge_accounting_mode(git_sha: str) -> str:
    if _uses_unique_edge_accounting(git_sha):
        return "unique physical"
    if not git_sha:
        return "legacy directed"
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BIDIRECTIONAL_EDGE_RESERVATION_COMMIT,
            git_sha,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return (
        "paired arcs normalized"
        if result.returncode == 0 else
        "legacy directed"
    )


def _physical_edge_overuse(
    item: Dict[str, Any],
    *,
    edge_accounting_mode: str,
) -> int:
    edge = int(item.get("edge_overuse", item.get("overuse_total", 0)))
    if edge_accounting_mode == "paired arcs normalized":
        if edge % 2:
            raise ValueError(
                f"directed-arc edge overuse must be paired, got {edge}"
            )
        edge //= 2
    return (
        edge
        + int(item.get("via_column_overuse", 0))
        + int(item.get("via_segment_overuse", 0))
    )


def _normalize_iteration(
    item: Dict[str, Any],
    *,
    edge_accounting_mode: str,
) -> Dict[str, Any]:
    normalized = dict(item)
    physical_edge = _physical_edge_overuse(
        item,
        edge_accounting_mode=edge_accounting_mode,
    )
    normalized["_physical_edge_overuse"] = physical_edge
    if "path_node_overuse_total" in item:
        normalized["_physical_negotiated_overuse"] = (
            physical_edge + int(item["path_node_overuse_total"])
        )
    elif "negotiated_overuse_total" in item:
        node_component = (
            int(item["negotiated_overuse_total"])
            - int(item.get("overuse_total", 0))
        )
        normalized["_physical_negotiated_overuse"] = (
            physical_edge + node_component
        )
    return normalized


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
        edge_accounting_mode = _edge_accounting_mode(
            str(journal.get("git_sha", ""))
        )
        iterations = [
            _normalize_iteration(
                item,
                edge_accounting_mode=edge_accounting_mode,
            )
            for item in journal.get("iterations", [])
        ]
        if not iterations and journal.get("status") != "starting":
            continue
        runs.append({
            "path": path,
            "journal": journal,
            "iterations": iterations,
            "edge_accounting_mode": edge_accounting_mode,
            "label": _label(path, journal),
        })
    return runs


def _row(run: Dict[str, Any]) -> Dict[str, Any]:
    journal = run["journal"]
    edge_accounting_mode = run.get(
        "edge_accounting_mode", "unique physical"
    )
    iterations = [
        (
            item
            if "_physical_edge_overuse" in item else
            _normalize_iteration(
                item,
                edge_accounting_mode=edge_accounting_mode,
            )
        )
        for item in run["iterations"]
    ]
    final = iterations[-1] if iterations else {}
    overuse = [
        int(item.get(
            "_physical_edge_overuse",
            _physical_edge_overuse(
                item,
                edge_accounting_mode=edge_accounting_mode,
            ),
        ))
        for item in iterations
    ]
    negotiated = [
        int(item["_physical_negotiated_overuse"])
        for item in iterations
        if "_physical_negotiated_overuse" in item
    ]
    barrels = [
        int(item.get("barrel_conflicts", 0)) for item in iterations
    ]
    path_nodes = [
        int(item.get("path_node_conflicts", 0))
        for item in iterations
    ]
    best_overuse = (
        min(
            iterations,
            key=lambda item: int(item.get(
                "_physical_edge_overuse",
                _physical_edge_overuse(
                    item,
                    edge_accounting_mode=edge_accounting_mode,
                ),
            )),
        )
        if iterations else {}
    )
    negotiated_iterations = [
        item for item in iterations
        if "_physical_negotiated_overuse" in item
    ]
    best_negotiated = (
        min(
            negotiated_iterations,
            key=lambda item: int(item["_physical_negotiated_overuse"]),
        )
        if negotiated_iterations else {}
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
        "edge_accounting": edge_accounting_mode,
        "initial_overuse": overuse[0] if overuse else "",
        "best_overuse": min(overuse) if overuse else "",
        "best_overuse_iteration": best_overuse.get("iteration", ""),
        "final_overuse": overuse[-1] if overuse else "",
        "initial_negotiated_overuse": (
            negotiated[0] if negotiated else ""
        ),
        "best_negotiated_overuse": (
            min(negotiated) if negotiated else ""
        ),
        "best_negotiated_overuse_iteration": best_negotiated.get(
            "iteration", ""
        ),
        "final_negotiated_overuse": (
            negotiated[-1] if negotiated else ""
        ),
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


def _stall_row(
    run: Dict[str, Any],
    plateau_window: int = STALL_WINDOW,
) -> Optional[Dict[str, Any]]:
    """Summarize a comparable full-board negotiated-overuse plateau."""
    if "8,192 nets" not in run["label"]:
        return None
    values = [
        (
            int(item["iteration"]),
            int(item["_physical_negotiated_overuse"]),
        )
        for item in run["iterations"]
        if "_physical_negotiated_overuse" in item
    ]
    if not values:
        return None
    layer_match = re.search(r"Backplane-(\d+)L", str(
        run["journal"].get("run_name", "")
    ))
    if not layer_match:
        return None
    best_position = min(
        range(len(values)),
        key=lambda index: values[index][1],
    )
    best_iteration, best_overuse = values[best_position]
    stalled = (
        len(values) - 1 - best_position >= max(1, plateau_window)
    )
    status = str(run["journal"].get("status", "unknown"))
    if stalled:
        observation = "stalled"
        stall_iteration = values[best_position + plateau_window][0]
        overuse_at_stall = best_overuse
    elif status == "routing":
        observation = "live_not_stalled"
        stall_iteration = ""
        overuse_at_stall = ""
    else:
        observation = "terminal_without_plateau"
        stall_iteration = ""
        overuse_at_stall = ""
    return {
        "layers": int(layer_match.group(1)),
        "run": run["label"],
        "git_sha": str(run["journal"].get("git_sha", "")),
        "status": status,
        "edge_accounting": run["edge_accounting_mode"],
        "iterations": len(values),
        "observation": observation,
        "stall_window": plateau_window,
        "stall_iteration": stall_iteration,
        "overuse_at_stall": overuse_at_stall,
        "best_iteration": best_iteration,
        "best_overuse": best_overuse,
        "current_overuse": values[-1][1],
        "eligible_for_fit": stalled,
        "progress_file": str(run["path"]),
    }


def _accounting_rank(row: Dict[str, Any]) -> int:
    return {
        "unique physical": 2,
        "paired arcs normalized": 1,
        "legacy directed": 0,
    }.get(str(row.get("edge_accounting", "")), -1)


def _selected_stalls(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Choose the newest best-accounted stalled run per layer."""
    selected: Dict[int, Tuple[int, int, Dict[str, Any]]] = {}
    for order, row in enumerate(rows):
        if not row.get("eligible_for_fit"):
            continue
        layers = int(row["layers"])
        candidate = (_accounting_rank(row), order, row)
        previous = selected.get(layers)
        if previous is None or candidate[:2] > previous[:2]:
            selected[layers] = candidate
    return [
        selected[layers][2] for layers in sorted(selected)
    ]


def _linear_layer_fit(
    rows: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Fit stalled negotiated overuse versus total copper layer count."""
    selected = _selected_stalls(rows)
    if len(selected) < 2:
        return None
    xs = [float(row["layers"]) for row in selected]
    ys = [float(row["overuse_at_stall"]) for row in selected]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        return None
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum(
        (y - (slope * x + intercept)) ** 2
        for x, y in zip(xs, ys)
    )
    total = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1.0 if total == 0 else 1.0 - residual / total
    zero_layer = -intercept / slope if slope < 0 else math.nan
    return {
        "points": float(len(selected)),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "zero_layer": zero_layer,
    }


def _write_stall_csv(
    rows: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_stall_markdown(
    rows: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    columns = (
        "layers",
        "run",
        "status",
        "edge_accounting",
        "iterations",
        "observation",
        "stall_iteration",
        "overuse_at_stall",
        "best_iteration",
        "best_overuse",
        "current_overuse",
        "eligible_for_fit",
    )
    lines = [
        "# Full-board layer/stall congestion",
        "",
        (
            f"A stall means no new minimum in the complete normalized "
            f"PathFinder objective for {STALL_WINDOW} consecutive "
            "iterations. The objective is physical edge/via-pool excess "
            "plus capacity-one graph-node excess; it is not a KiCad DRC "
            "count."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns)
            + " |"
        )
    fit = _linear_layer_fit(rows)
    lines.extend(["", "## Extrapolation", ""])
    if fit is None:
        lines.append(
            "Withheld: at least two distinct total-layer counts with "
            "comparable full-board stalls are required. A single-layer "
            "history cannot identify a layer-capacity slope."
        )
    else:
        zero = fit["zero_layer"]
        zero_text = (
            f"{zero:.2f} layers"
            if math.isfinite(zero) else
            "not projected because the fitted slope is non-negative"
        )
        lines.append(
            f"Ordinary least-squares fit over {int(fit['points'])} "
            f"selected layer counts: slope {fit['slope']:.1f} excess "
            f"uses/layer, R²={fit['r_squared']:.3f}, zero intercept "
            f"{zero_text}. This is a planning extrapolation, not a "
            "routability proof."
        )
    lines.extend([
        "",
        "For each layer count, the fit prefers unique-physical accounting, "
        "then paired-arc-normalized accounting, and uses the newest "
        "available stalled run at that accounting quality.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_stall_svg(
    rows: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    width, height = 1200, 820
    plot_x, plot_y, plot_width, plot_height = 110, 120, 980, 540
    layer_values = [14, 16, 18, 20, 24, 28, 32]
    if rows:
        layer_values.extend(int(row["layers"]) for row in rows)
    min_layer = min(layer_values)
    max_layer = max(layer_values)
    y_values = [
        int(row["overuse_at_stall"] or row["best_overuse"])
        for row in rows
    ]
    y_max = max(y_values, default=1)
    y_max = max(1, int(math.ceil(y_max * 1.10 / 10_000) * 10_000))

    def x_at(layers: float) -> float:
        return plot_x + plot_width * (
            (layers - min_layer) / max(1, max_layer - min_layer)
        )

    def y_at(value: float) -> float:
        return plot_y + plot_height * (1.0 - value / y_max)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="60" y="42" font-family="sans-serif" font-size="25" '
        'font-weight="bold">Full-board plateau congestion vs copper '
        'layers</text>',
        '<text x="60" y="72" font-family="sans-serif" font-size="14" '
        'fill="#475569">8,192 nets · 0.4 mm grid · physical edge/via '
        '+ exact node excess · linear scale to zero</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" '
        f'height="{plot_height}" fill="#fafafa" stroke="#94a3b8"/>',
    ]
    for tick in range(6):
        value = y_max * (1.0 - tick / 5)
        y = plot_y + plot_height * tick / 5
        svg.extend([
            f'<line x1="{plot_x}" y1="{y:.1f}" '
            f'x2="{plot_x + plot_width}" y2="{y:.1f}" '
            'stroke="#e2e8f0"/>',
            f'<text x="{plot_x - 10}" y="{y + 4:.1f}" '
            'text-anchor="end" font-family="monospace" font-size="12">'
            f'{int(value):,}</text>',
        ])
    for layers in range(min_layer, max_layer + 1, 2):
        x = x_at(layers)
        svg.extend([
            f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" '
            f'y2="{plot_y + plot_height}" stroke="#f1f5f9"/>',
            f'<text x="{x:.1f}" y="{plot_y + plot_height + 24}" '
            'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'{layers}</text>',
        ])
    svg.extend([
        f'<text x="{plot_x + plot_width / 2:.1f}" '
        f'y="{plot_y + plot_height + 56}" text-anchor="middle" '
        'font-family="sans-serif" font-size="14">Total copper layers</text>',
        f'<text x="25" y="{plot_y + plot_height / 2:.1f}" '
        'transform="rotate(-90 25 '
        f'{plot_y + plot_height / 2:.1f})" text-anchor="middle" '
        'font-family="sans-serif" font-size="14">'
        'Negotiated excess uses at plateau</text>',
    ])

    selected = _selected_stalls(rows)
    fit = _linear_layer_fit(rows)
    if fit is not None:
        x1, x2 = float(min_layer), float(max_layer)
        y1 = max(0.0, fit["slope"] * x1 + fit["intercept"])
        y2 = max(0.0, fit["slope"] * x2 + fit["intercept"])
        svg.append(
            f'<line x1="{x_at(x1):.1f}" y1="{y_at(y1):.1f}" '
            f'x2="{x_at(x2):.1f}" y2="{y_at(y2):.1f}" '
            'stroke="#0f172a" stroke-width="2" stroke-dasharray="8 6"/>'
        )

    for index, row in enumerate(rows):
        value = int(row["overuse_at_stall"] or row["best_overuse"])
        x = x_at(float(row["layers"]))
        x += ((index % 5) - 2) * 5
        y = y_at(value)
        selected_point = row in selected
        color = "#dc2626" if row["eligible_for_fit"] else "#2563eb"
        if row["eligible_for_fit"]:
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" '
                f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
        else:
            svg.append(
                f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" '
                f'height="12" fill="#ffffff" stroke="{color}" '
                'stroke-width="3"/>'
            )
        weight = "bold" if selected_point else "normal"
        state = (
            "stall" if row["eligible_for_fit"] else "live best"
        )
        short_label = (
            f"{row['layers']}L {str(row['git_sha'])[:7]} "
            f"{state}: {value:,}"
        )
        label_y = y - 10 - (index % 4) * 18
        svg.append(
            f'<text x="{x + 10:.1f}" y="{label_y:.1f}" '
            f'font-family="sans-serif" font-size="11" '
            f'font-weight="{weight}">{html.escape(short_label)}</text>'
        )
    if fit is None:
        fit_text = (
            "Trend withheld: need comparable stalls at ≥2 distinct "
            "layer counts"
        )
    else:
        zero = fit["zero_layer"]
        zero_text = (
            f"zero intercept ≈ {zero:.2f} layers"
            if math.isfinite(zero) else
            "no finite zero intercept"
        )
        fit_text = (
            f"OLS slope {fit['slope']:.1f} excess/layer; "
            f"R² {fit['r_squared']:.3f}; {zero_text}"
        )
    svg.extend([
        f'<text x="{plot_x}" y="735" font-family="sans-serif" '
        f'font-size="14" font-weight="bold">{html.escape(fit_text)}</text>',
        f'<circle cx="{plot_x}" cy="775" r="7" fill="#dc2626"/>',
        f'<text x="{plot_x + 14}" y="779" font-family="sans-serif" '
        'font-size="12">observed ≥8-iteration stall</text>',
        f'<rect x="{plot_x + 310}" y="769" width="12" height="12" '
        'fill="#ffffff" stroke="#2563eb" stroke-width="3"/>',
        f'<text x="{plot_x + 330}" y="779" font-family="sans-serif" '
        'font-size="12">live/terminal point without observed stall</text>',
        "</svg>",
    ])
    path.write_text("\n".join(svg), encoding="utf-8", newline="\n")


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    columns = (
        "run", "status", "iterations", "routed_nets", "target_nets",
        "complete", "edge_accounting",
        "initial_overuse", "best_overuse", "final_overuse",
        "best_overuse_iteration", "initial_physical", "best_physical",
        "best_physical_iteration", "final_physical",
        "initial_path_nodes", "best_path_nodes",
        "best_path_nodes_iteration", "final_path_nodes",
        "initial_negotiated_overuse", "best_negotiated_overuse",
        "best_negotiated_overuse_iteration",
        "final_negotiated_overuse",
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
    width, height = 1200, 1660
    plot_x, plot_width = 90, 1040
    panel_height = 200
    panels = [
        (
            "Unique physical edge/via overuse (log scale)",
            "_physical_edge_overuse",
            90,
            True,
        ),
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
        (
            "Negotiated edge + exact node excess (log scale)",
            "_physical_negotiated_overuse",
            990,
            True,
        ),
        ("Present congestion pressure", "pres_fac", 1290, False),
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
                float(item[key])
                for run in runs
                for item in run["iterations"]
                if key in item
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
                float(item[key]) for item in run["iterations"]
                if key in item
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
    legend_y = 1570
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
    stall_rows = [
        row for row in (_stall_row(run) for run in runs)
        if row is not None
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.output_dir / "reduced-layer-comparison.csv")
    _write_markdown(rows, args.output_dir / "reduced-layer-comparison.md")
    _write_svg(runs, args.output_dir / "reduced-layer-progress.svg")
    _write_stall_csv(
        stall_rows,
        args.output_dir / "layer-stall-overuse.csv",
    )
    _write_stall_markdown(
        stall_rows,
        args.output_dir / "layer-stall-overuse.md",
    )
    _write_stall_svg(
        stall_rows,
        args.output_dir / "layer-stall-overuse.svg",
    )
    print(json.dumps({
        "runs": len(runs),
        "csv": str(args.output_dir / "reduced-layer-comparison.csv"),
        "markdown": str(args.output_dir / "reduced-layer-comparison.md"),
        "svg": str(args.output_dir / "reduced-layer-progress.svg"),
        "stall_csv": str(args.output_dir / "layer-stall-overuse.csv"),
        "stall_markdown": str(
            args.output_dir / "layer-stall-overuse.md"
        ),
        "stall_svg": str(args.output_dir / "layer-stall-overuse.svg"),
    }, indent=2))


if __name__ == "__main__":
    main()
