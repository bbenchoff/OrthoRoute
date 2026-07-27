"""Write compact CSV and Markdown tables from a KiCad JSON DRC report."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


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
    return {
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "rule_errors": rule_errors,
        "unconnected_items": unconnected,
        "reported_errors": rule_errors + unconnected,
        "warnings": warnings,
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
