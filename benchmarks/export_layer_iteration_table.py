"""Export every PCBWay monster-route iteration into root CSV/Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "LAYER_ITERATION_TABLE.csv"
DEFAULT_MARKDOWN = ROOT / "LAYER_ITERATION_TABLE.md"

CSV_FIELDS = [
    "total_layers",
    "internal_layers",
    "scope",
    "run_kind",
    "run_started",
    "run_name",
    "git_sha",
    "status",
    "terminal_reason",
    "iteration",
    "routed_nets",
    "failed_nets",
    "elapsed_s",
    "complete_excess",
    "delta_complete_excess",
    "running_best_excess",
    "improvement_from_start_pct",
    "overuse_per_routed_net",
    "edge_via_excess",
    "edge_excess",
    "via_column_excess",
    "via_segment_excess",
    "node_excess",
    "node_share_pct",
    "exact_barrel_conflicts",
    "portal_grid_conflicts",
    "escape_conflicts",
    "pressure",
    "pressure_ceiling",
    "adaptive_pressure_limit",
    "hotset_size",
    "hotset_cap",
    "hotset_conflict_coverage_pct",
    "slow_events",
    "rolling_5_improvement_pct",
    "pressure_reference_ceiling",
    "pressure_reference_improvement_pct",
    "pressure_underperform_windows",
    "pressure_backoffs",
    "pressure_rejected_ceiling",
    "hottest_node_layer",
    "hottest_node_layer_excess",
    "most_occupied_layer",
    "peak_node_occupancy_pct",
    "grid_pitch_mm",
    "direction_mode",
    "depth_bias",
    "fabrication_profile",
    "hdi_stack",
    "source_sha256",
    "journal",
]

MARKDOWN_FIELDS = [
    ("iteration", "iter"),
    ("routed_nets", "nets"),
    ("complete_excess", "complete excess"),
    ("delta_complete_excess", "delta"),
    ("running_best_excess", "best"),
    ("edge_via_excess", "edge/via"),
    ("node_excess", "node"),
    ("exact_barrel_conflicts", "barrels"),
    ("portal_grid_conflicts", "portals"),
    ("escape_conflicts", "escapes"),
    ("pressure", "pressure"),
    ("pressure_ceiling", "ceiling"),
    ("hotset_size", "hotset"),
    ("slow_events", "slow"),
    ("rolling_5_improvement_pct", "5-pass %"),
    ("hottest_node_layer", "hot layer"),
    ("hottest_node_layer_excess", "hot excess"),
    ("elapsed_s", "elapsed s"),
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    return None if number is None else int(number)


def _rounded(value: Any, digits: int = 3) -> Any:
    number = _number(value)
    return "" if number is None else round(number, digits)


def _layers(journal: Dict[str, Any]) -> Optional[int]:
    config = journal.get("experiment_config") or {}
    configured = _integer(config.get("layer_count"))
    if configured:
        return configured
    match = re.search(r"Backplane-(\d+)L", str(journal.get("run_name", "")))
    return int(match.group(1)) if match else None


def _campaign_journal(journal: Dict[str, Any]) -> bool:
    config = journal.get("experiment_config") or {}
    run_name = str(journal.get("run_name", ""))
    profile = (
        config.get("fabrication_profile")
        or journal.get("fabrication_profile")
        or ""
    )
    stack = config.get("hdi_stack") or ""
    grid = _number(config.get("grid_pitch"))
    tagged = (
        "FAB-pcbway_advanced_hdi" in run_name
        and "HDI-pcbway_mechanical" in run_name
    )
    explicit = (
        profile == "pcbway_advanced_hdi"
        and stack == "pcbway_mechanical"
        and (grid is None or abs(grid - 0.4) < 1e-9)
    )
    return bool(tagged or explicit)


def _net_limit(journal: Dict[str, Any]) -> int:
    config = journal.get("experiment_config") or {}
    configured = _integer(config.get("net_limit")) or 0
    if configured:
        return configured
    match = re.search(
        r"-(\d+)N-",
        str(journal.get("run_name", "")),
    )
    return int(match.group(1)) if match else 0


def _run_kind(journal: Dict[str, Any]) -> str:
    limit = _net_limit(journal)
    warm = journal.get("warm_start")
    base = "peel" if isinstance(warm, dict) and warm else "fresh"
    return f"preview-{base}" if limit else base


def _scope(journal: Dict[str, Any]) -> str:
    limit = _net_limit(journal)
    return f"{limit}-net-preview" if limit else "full-board"


def _layer_extrema(
    layer_rows: Any,
) -> tuple[Any, Any, Any, Any]:
    if not isinstance(layer_rows, list) or not layer_rows:
        return "", "", "", ""
    rows = [row for row in layer_rows if isinstance(row, dict)]
    if not rows:
        return "", "", "", ""
    hottest = max(
        rows,
        key=lambda row: _number(row.get("excess_uses")) or 0.0,
    )
    occupied = max(
        rows,
        key=lambda row: _number(row.get("occupied_nodes")) or 0.0,
    )
    capacity = _number(occupied.get("capacity_nodes"))
    usage = _number(occupied.get("occupied_nodes"))
    occupancy_pct = (
        ""
        if not capacity or usage is None
        else round(100.0 * usage / capacity, 3)
    )
    return (
        _integer(hottest.get("layer")),
        _integer(hottest.get("excess_uses")),
        _integer(occupied.get("layer")),
        occupancy_pct,
    )


def rows_from_journal(
    journal: Dict[str, Any],
    journal_path: Path,
) -> List[Dict[str, Any]]:
    layers = _layers(journal)
    if layers is None:
        return []
    config = journal.get("experiment_config") or {}
    terminal_reason = (
        journal.get("termination_reason")
        or journal.get("error")
        or ""
    )
    iterations = journal.get("iterations")
    if not isinstance(iterations, list):
        return []

    rows: List[Dict[str, Any]] = []
    first_excess: Optional[float] = None
    previous_excess: Optional[float] = None
    running_best: Optional[float] = None
    for iteration in iterations:
        if not isinstance(iteration, dict):
            continue
        complete = _number(iteration.get("negotiated_overuse_total"))
        edge_via = _number(iteration.get("overuse_total"))
        node = _number(iteration.get("path_node_overuse_total"))
        if complete is None and edge_via is not None:
            complete = edge_via + (node or 0.0)
        if complete is not None:
            if first_excess is None:
                first_excess = complete
            running_best = (
                complete
                if running_best is None
                else min(running_best, complete)
            )
        routed_nets = _integer(iteration.get("routed_nets"))
        delta = (
            ""
            if complete is None or previous_excess is None
            else int(complete - previous_excess)
        )
        start_improvement = (
            ""
            if complete is None or not first_excess
            else round(100.0 * (first_excess - complete) / first_excess, 3)
        )
        overuse_per_net = (
            ""
            if complete is None or not routed_nets
            else round(complete / routed_nets, 6)
        )
        node_share = (
            ""
            if not complete or node is None
            else round(100.0 * node / complete, 3)
        )
        hot_layer, hot_excess, occupied_layer, occupancy_pct = (
            _layer_extrema(iteration.get("path_node_layers"))
        )
        row = {
            "total_layers": layers,
            "internal_layers": max(0, layers - 2),
            "scope": _scope(journal),
            "run_kind": _run_kind(journal),
            "run_started": journal.get("started", ""),
            "run_name": journal.get("run_name", journal_path.stem),
            "git_sha": journal.get("git_sha", ""),
            "status": journal.get("status", ""),
            "terminal_reason": terminal_reason,
            "iteration": _integer(iteration.get("iteration")),
            "routed_nets": routed_nets,
            "failed_nets": (
                _integer(iteration.get("failed_nets"))
                if iteration.get("failed_nets") is not None
                else _integer(iteration.get("failed"))
            ),
            "elapsed_s": _rounded(iteration.get("elapsed_seconds")),
            "complete_excess": (
                "" if complete is None else int(complete)
            ),
            "delta_complete_excess": delta,
            "running_best_excess": (
                "" if running_best is None else int(running_best)
            ),
            "improvement_from_start_pct": start_improvement,
            "overuse_per_routed_net": overuse_per_net,
            "edge_via_excess": (
                "" if edge_via is None else int(edge_via)
            ),
            "edge_excess": _integer(iteration.get("edge_overuse")),
            "via_column_excess": _integer(
                iteration.get("via_column_overuse")
            ),
            "via_segment_excess": _integer(
                iteration.get("via_segment_overuse")
            ),
            "node_excess": "" if node is None else int(node),
            "node_share_pct": node_share,
            "exact_barrel_conflicts": _integer(
                iteration.get("exact_barrel_conflicts")
            ),
            "portal_grid_conflicts": _integer(
                iteration.get("portal_grid_conflicts")
            ),
            "escape_conflicts": _integer(
                iteration.get("escape_conflicts")
            ),
            "pressure": _rounded(iteration.get("pres_fac")),
            "pressure_ceiling": _rounded(
                iteration.get("pres_fac_max")
            ),
            "adaptive_pressure_limit": _rounded(
                iteration.get("adaptive_pressure_limit")
            ),
            "hotset_size": _integer(iteration.get("hotset_size")),
            "hotset_cap": _integer(iteration.get("hotset_cap")),
            "hotset_conflict_coverage_pct": (
                ""
                if _number(
                    iteration.get(
                        "hotset_conflict_pair_coverage_fraction"
                    )
                )
                is None
                else round(
                    100.0
                    * float(
                        iteration[
                            "hotset_conflict_pair_coverage_fraction"
                        ]
                    ),
                    3,
                )
            ),
            "slow_events": _integer(
                iteration.get("slow_progress_events")
            ),
            "rolling_5_improvement_pct": (
                ""
                if _number(iteration.get("slow_progress_fraction")) is None
                else round(
                    100.0
                    * float(iteration["slow_progress_fraction"]),
                    3,
                )
            ),
            "pressure_reference_ceiling": _rounded(
                iteration.get("pressure_trial_reference_ceiling")
            ),
            "pressure_reference_improvement_pct": (
                ""
                if _number(
                    iteration.get("pressure_trial_reference_fraction")
                )
                is None
                else round(
                    100.0
                    * float(
                        iteration["pressure_trial_reference_fraction"]
                    ),
                    3,
                )
            ),
            "pressure_underperform_windows": _integer(
                iteration.get("pressure_trial_underperform_count")
            ),
            "pressure_backoffs": _integer(
                iteration.get("pressure_backoff_count")
            ),
            "pressure_rejected_ceiling": _rounded(
                iteration.get("pressure_rejected_ceiling")
            ),
            "hottest_node_layer": hot_layer,
            "hottest_node_layer_excess": hot_excess,
            "most_occupied_layer": occupied_layer,
            "peak_node_occupancy_pct": occupancy_pct,
            "grid_pitch_mm": _rounded(config.get("grid_pitch")),
            "direction_mode": config.get("direction_mode", ""),
            "depth_bias": _rounded(config.get("layer_depth_bias")),
            "fabrication_profile": (
                config.get("fabrication_profile")
                or journal.get("fabrication_profile", "")
            ),
            "hdi_stack": config.get("hdi_stack", ""),
            "source_sha256": journal.get("source_sha256", ""),
            "journal": str(journal_path),
        }
        rows.append(row)
        if complete is not None:
            previous_excess = complete
    return rows


def collect_rows(
    results_dir: Path,
    *,
    campaign_only: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*progress.json")):
        journal = _read_json(path)
        if journal is None:
            continue
        if campaign_only and not _campaign_journal(journal):
            continue
        rows.extend(rows_from_journal(journal, path))
    rows.sort(
        key=lambda row: (
            int(row["total_layers"]),
            str(row["run_started"]),
            str(row["run_name"]),
            int(row["iteration"] or 0),
        )
    )
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _md(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _run_heading(row: Dict[str, Any]) -> str:
    return (
        f"{row['run_started']} -- {row['run_kind']} -- "
        f"{row['scope']} -- `{row['status']}`"
    )


def write_markdown(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    *,
    source_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    layer_counts = sorted({int(row["total_layers"]) for row in rows})
    run_count = len({str(row["journal"]) for row in rows})
    lines = [
        "# OrthoRoute layer-by-iteration table",
        "",
        f"Generated: `{generated}`",
        "",
        f"Source directory: `{source_dir}`",
        "",
        (
            f"Rows: **{len(rows):,}**; runs: **{run_count:,}**; "
            f"layer counts: **{', '.join(map(str, layer_counts))}**."
        ),
        "",
        (
            "The companion `LAYER_ITERATION_TABLE.csv` contains the full "
            "sortable schema, including pressure trials, layer hotspots, "
            "fabrication settings, source hashes, and journal paths."
        ),
        "",
    ]
    current_layer: Optional[int] = None
    current_run: Optional[str] = None
    for row in rows:
        layer = int(row["total_layers"])
        run_name = str(row["run_name"])
        if layer != current_layer:
            current_layer = layer
            current_run = None
            lines.extend([f"## {layer} total copper layers", ""])
        if run_name != current_run:
            if current_run is not None:
                lines.append("")
            current_run = run_name
            lines.extend(
                [
                    f"### {_run_heading(row)}",
                    "",
                    f"Run: `{run_name}`",
                    "",
                ]
            )
            if row.get("terminal_reason"):
                lines.extend(
                    [
                        f"Terminal reason: `{_md(row['terminal_reason'])}`",
                        "",
                    ]
                )
            headers = [label for _, label in MARKDOWN_FIELDS]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
        values = [_md(row.get(key, "")) for key, _ in MARKDOWN_FIELDS]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(path)


def export_tables(
    results_dir: Path,
    csv_path: Path,
    markdown_path: Path,
    *,
    campaign_only: bool = True,
) -> List[Dict[str, Any]]:
    rows = collect_rows(results_dir, campaign_only=campaign_only)
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, source_dir=results_dir)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--all-journals",
        action="store_true",
        help="include non-PCBWay and non-0.4 mm progress journals",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    rows = export_tables(
        args.results_dir.resolve(),
        args.csv.resolve(),
        args.markdown.resolve(),
        campaign_only=not args.all_journals,
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "runs": len({row["journal"] for row in rows}),
                "layers": sorted(
                    {int(row["total_layers"]) for row in rows}
                ),
                "csv": str(args.csv.resolve()),
                "markdown": str(args.markdown.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
