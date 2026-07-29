"""Write compact CSV and Markdown tables from a KiCad JSON DRC report."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFECT_SITE_TYPES = {
    "clearance",
    "hole_clearance",
    "shorting_items",
    "tracks_crossing",
}


def summarize(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    counts = Counter(
        ("violation", item.get("severity", "unknown"), item.get(
            "type", "unknown"
        ))
        for item in report.get("violations", [])
    )
    counts.update(
        ("unconnected", item.get("severity", "error"), item.get(
            "type", "unconnected_items"
        ))
        for item in report.get("unconnected_items", [])
    )
    return [
        {
            "section": section,
            "severity": severity,
            "type": check_type,
            "count": count,
        }
        for (section, severity, check_type), count in sorted(
            counts.items(),
            key=lambda entry: (
                entry[0][1] != "error",
                -entry[1],
                entry[0],
            ),
        )
    ]


def defect_site_census(
    report: Dict[str, Any],
    *,
    merge_radius_mm: float = 0.25,
) -> List[Dict[str, Any]]:
    """Cluster co-located electrical DRC reports into physical defect sites.

    KiCad commonly emits a clearance, short, and/or hole-clearance report for
    one physical conflict.  A 0.25 mm radius joins those duplicate reports
    without joining neighboring sites on the 0.4 mm routing lattice.
    """
    defects = []
    for item in report.get("violations", []):
        if (
            item.get("severity") != "error"
            or item.get("type") not in DEFECT_SITE_TYPES
        ):
            continue
        positions = [
            child.get("pos")
            for child in item.get("items", [])
            if isinstance(child.get("pos"), dict)
            and child["pos"].get("x") is not None
            and child["pos"].get("y") is not None
        ]
        if positions:
            x = sum(float(pos["x"]) for pos in positions) / len(positions)
            y = sum(float(pos["y"]) for pos in positions) / len(positions)
        else:
            x = y = None
        defects.append({
            "type": str(item.get("type", "unknown")),
            "description": str(item.get("description", "")),
            "x_mm": x,
            "y_mm": y,
        })

    parents = list(range(len(defects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    cell_size = merge_radius_mm
    cells: Dict[tuple, List[int]] = {}
    for index, defect in enumerate(defects):
        if defect["x_mm"] is None:
            continue
        cell = (
            math.floor(defect["x_mm"] / cell_size),
            math.floor(defect["y_mm"] / cell_size),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), []):
                    other_defect = defects[other]
                    distance = math.hypot(
                        defect["x_mm"] - other_defect["x_mm"],
                        defect["y_mm"] - other_defect["y_mm"],
                    )
                    if distance <= merge_radius_mm:
                        union(index, other)
        cells.setdefault(cell, []).append(index)

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for index, defect in enumerate(defects):
        groups.setdefault(find(index), []).append(defect)

    def group_position(group):
        positioned = [item for item in group if item["x_mm"] is not None]
        if not positioned:
            return (float("inf"), float("inf"))
        return (
            sum(item["x_mm"] for item in positioned) / len(positioned),
            sum(item["y_mm"] for item in positioned) / len(positioned),
        )

    rows = []
    ordered = sorted(groups.values(), key=group_position)
    for site_number, group in enumerate(ordered, start=1):
        positioned = [item for item in group if item["x_mm"] is not None]
        x = (
            sum(item["x_mm"] for item in positioned) / len(positioned)
            if positioned else None
        )
        y = (
            sum(item["y_mm"] for item in positioned) / len(positioned)
            if positioned else None
        )
        type_counts = Counter(item["type"] for item in group)
        rows.append({
            "site_id": f"S{site_number:04d}",
            "x_mm": None if x is None else round(x, 6),
            "y_mm": None if y is None else round(y, 6),
            "report_count": len(group),
            "report_types": ";".join(
                f"{name}:{count}" for name, count in sorted(
                    type_counts.items()
                )
            ),
        })
    return rows


def write_defect_site_census(
    report: Dict[str, Any],
    output_stem: Path,
) -> Dict[str, Any]:
    rows = defect_site_census(report)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    markdown_path = output_stem.with_suffix(".md")
    payload = {
        "merge_radius_mm": 0.25,
        "included_types": sorted(DEFECT_SITE_TYPES),
        "report_count": sum(row["report_count"] for row in rows),
        "defect_site_count": len(rows),
        "sites": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "site_id",
                "x_mm",
                "y_mm",
                "report_count",
                "report_types",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Deduplicated KiCad defect-site census",
        "",
        f"- Physical defect sites: {len(rows)}",
        f"- Clustered DRC reports: {payload['report_count']}",
        "- Merge radius: 0.25 mm",
        "",
        "| site | x (mm) | y (mm) | reports | types |",
        "|---|---:|---:|---:|---|",
    ]
    lines.extend(
        "| {site_id} | {x_mm} | {y_mm} | {report_count} | "
        "{report_types} |".format(**row)
        for row in rows
    )
    markdown_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "defect_site_count": len(rows),
        "clustered_report_count": payload["report_count"],
    }


def write_summary(
    report_path: Path,
    output_stem: Path,
) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = summarize(report)
    csv_path = output_stem.with_suffix(".csv")
    markdown_path = output_stem.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("section", "severity", "type", "count"),
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# KiCad DRC summary",
        "",
        "| section | severity | type | count |",
        "|---|---|---|---:|",
    ]
    lines.extend(
        "| {section} | {severity} | {type} | {count} |".format(**row)
        for row in rows
    )
    rule_errors = sum(
        row["count"]
        for row in rows
        if row["section"] == "violation" and row["severity"] == "error"
    )
    unconnected = sum(
        row["count"] for row in rows if row["section"] == "unconnected"
    )
    warnings = sum(
        row["count"] for row in rows if row["severity"] == "warning"
    )
    lines.extend([
        "",
        f"- Rule errors: {rule_errors}",
        f"- Unconnected items: {unconnected}",
        f"- Reported electrical errors: {rule_errors + unconnected}",
        f"- Warnings: {warnings}",
        "",
    ])
    markdown_path.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    census_name = (
        output_stem.name[:-len("-summary")] + "-defect-sites"
        if output_stem.name.endswith("-summary")
        else output_stem.name + "-defect-sites"
    )
    census = write_defect_site_census(
        report,
        output_stem.with_name(census_name),
    )
    return {
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "rule_errors": rule_errors,
        "unconnected_items": unconnected,
        "reported_errors": rule_errors + unconnected,
        "warnings": warnings,
        "defect_site_census": census,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-stem", type=Path)
    args = parser.parse_args()
    output_stem = args.output_stem or args.report.with_name(
        args.report.stem + "-summary"
    )
    print(json.dumps(write_summary(args.report, output_stem), indent=2))


if __name__ == "__main__":
    main()
