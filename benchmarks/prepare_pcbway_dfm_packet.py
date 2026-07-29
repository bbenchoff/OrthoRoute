"""Prepare an honest PCBWay DFM-review packet for an accepted board.

This does not assert manufacturability.  It packages the exact routed drill
schedule and highlights every point not clearly covered by PCBWay's published
capabilities so the fabricator can propose or approve the final stackup.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


PCBWAY_SOURCES = [
    {
        "title": "HDI PCB Manufacturing Capabilities",
        "url": "https://www.pcbway.com/hdi-pcb.html",
        "used_for": (
            "up to 60 layers; evaluation above 30; 0.15 mm minimum "
            "mechanical drill"
        ),
    },
    {
        "title": "Advanced PCB Manufacturing Capability",
        "url": "https://www.pcbway.com/advanced-pcb-capabilities.html",
        "used_for": (
            "2 mil minimum insulating layer; 3 mil minimum via annular "
            "ring; 0.15 mm mechanical hole"
        ),
    },
    {
        "title": "Mechanical Blind-Via Manufacturing",
        "url": (
            "https://www.pcbway.com/blog/Engineering_Technical/"
            "Key_Points_in_Manufacturing_Mechanical_Blind_Via.html"
        ),
        "used_for": (
            "published description of mechanical blind holes as generally "
            "passing through 3-4 circuit layers"
        ),
    },
    {
        "title": "Blind and Buried Via Design",
        "url": (
            "https://www.pcbway.com/pcb_prototype/"
            "Blind_vias_and_Buried_Vias.html"
        ),
        "used_for": (
            "lamination-dependent start/end rules and advice to obtain "
            "fabricator approval"
        ),
    },
]

COPPER_THICKNESS_MM = 0.035
MASK_THICKNESS_MM = 0.01
CENTRAL_CORE_MM = 0.19
PUBLISHED_MIN_DIELECTRIC_MM = 0.0508
PUBLISHED_MIN_MECHANICAL_DRILL_MM = 0.15
PUBLISHED_MIN_ANNULAR_RING_MM = 0.0762


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def provisional_stackup(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = list(manifest["copper_layers"])
    thickness = float(manifest["board_thickness_mm"])
    gap_count = len(layers) - 1
    prepreg_count = gap_count - 1
    prepreg = (
        thickness
        - 2 * MASK_THICKNESS_MM
        - len(layers) * COPPER_THICKNESS_MM
        - CENTRAL_CORE_MM
    ) / prepreg_count
    central_gap = len(layers) // 2 - 1
    rows = []
    for index, layer in enumerate(layers):
        rows.append({
            "sequence": len(rows) + 1,
            "name": layer,
            "kind": "copper",
            "thickness_mm": COPPER_THICKNESS_MM,
            "published_min_mm": "",
            "review": "provisional_1oz_copper",
        })
        if index == len(layers) - 1:
            continue
        is_core = index == central_gap
        dielectric = CENTRAL_CORE_MM if is_core else prepreg
        rows.append({
            "sequence": len(rows) + 1,
            "name": f"dielectric {index + 1}",
            "kind": "core" if is_core else "prepreg",
            "thickness_mm": round(dielectric, 9),
            "published_min_mm": PUBLISHED_MIN_DIELECTRIC_MM,
            "review": (
                "within_published_minimum"
                if dielectric >= PUBLISHED_MIN_DIELECTRIC_MM
                else "below_published_2mil_minimum"
            ),
        })
    return rows


def span_review(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for span in manifest.get("via_span_schedule", []):
        copper_span = int(span["copper_layers_spanned"])
        drill = float(span["drill_mm"])
        diameter = float(span["diameter_mm"])
        ring = (diameter - drill) / 2
        issues = []
        if copper_span > 4:
            issues.append("deeper_than_published_typical_mechanical_3-4L")
        if copper_span % 2:
            issues.append("odd_copper_layer_span_requires_stackup_review")
        if drill < PUBLISHED_MIN_MECHANICAL_DRILL_MM:
            issues.append("below_published_mechanical_drill_minimum")
        if ring < PUBLISHED_MIN_ANNULAR_RING_MM:
            issues.append("below_published_annular_ring_minimum")
        if int(span["dielectric_gaps_spanned"]) > 1:
            issues.append("nonadjacent_span_requires_lamination_mapping")
        if not issues:
            issues.append("still_requires_final_stackup_mapping")
        rows.append({
            **span,
            "annular_ring_mm": round(ring, 6),
            "review_status": (
                "not_clearly_covered"
                if any(
                    issue != "still_requires_final_stackup_mapping"
                    for issue in issues
                )
                else "requires_confirmation"
            ),
            "review_reasons": ";".join(issues),
        })
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_stackup_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Provisional stackup audit",
        "",
        "This reproduces the stackup currently embedded in the KiCad export. "
        "It is an audit input, not a PCBWay-approved construction.",
        "",
        "| seq | name | kind | thickness (mm) | published min (mm) | review |",
        "|---:|---|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['sequence']} | {row['name']} | {row['kind']} | "
        f"{row['thickness_mm']} | {row['published_min_mm']} | "
        f"{row['review']} |"
        for row in rows
    )
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_span_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    unclear = [
        row for row in rows
        if row["review_status"] == "not_clearly_covered"
    ]
    lines = [
        "# Via span review",
        "",
        f"- Span classes: {len(rows)}",
        f"- Classes not clearly covered: {len(unclear)}",
        f"- Total drill events: {sum(int(row['count']) for row in rows)}",
        "- This classification is deliberately conservative. PCBWay DFM "
        "approval, not this file, determines manufacturability.",
        "",
        "| from | to | copper layers | gaps | type | drill (mm) | count | "
        "review |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['from_layer']} | {row['to_layer']} | "
        f"{row['copper_layers_spanned']} | "
        f"{row['dielectric_gaps_spanned']} | {row['via_type']} | "
        f"{row['drill_mm']} | {row['count']} | "
        f"{row['review_reasons']} |"
        for row in rows
    )
    path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def prepare_packet(
    manifest_path: Path,
    drc_summary_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stackup = provisional_stackup(manifest)
    spans = span_review(manifest)
    unclear = [
        row for row in spans
        if row["review_status"] == "not_clearly_covered"
    ]
    stackup_csv = output_dir / "provisional-stackup.csv"
    stackup_md = output_dir / "provisional-stackup.md"
    span_csv = output_dir / "span-review.csv"
    span_md = output_dir / "span-review.md"
    sources_json = output_dir / "pcbway-published-sources.json"
    _write_csv(stackup_csv, stackup)
    _write_stackup_markdown(stackup_md, stackup)
    _write_csv(span_csv, spans)
    _write_span_markdown(span_md, spans)
    sources_json.write_text(
        json.dumps(PCBWAY_SOURCES, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    copied_manifest = output_dir / manifest_path.name
    copied_drc = output_dir / drc_summary_path.name
    shutil.copy2(manifest_path, copied_manifest)
    shutil.copy2(drc_summary_path, copied_drc)
    for suffix in (".csv", ".md"):
        sibling = manifest_path.with_suffix(suffix)
        if sibling.exists():
            shutil.copy2(sibling, output_dir / sibling.name)

    dielectric_violations = sum(
        row["review"] == "below_published_2mil_minimum"
        for row in stackup
    )
    total_drills = sum(int(row["count"]) for row in spans)
    nonadjacent_drills = sum(
        int(row["count"])
        for row in spans
        if int(row["dielectric_gaps_spanned"]) > 1
    )
    board = Path(manifest["board"])
    review = {
        "status": "prepared_for_pcbway_dfm_review_not_approved",
        "board": str(board),
        "layer_count": int(manifest["layer_count"]),
        "board_thickness_mm": float(manifest["board_thickness_mm"]),
        "total_drill_events": total_drills,
        "span_class_count": len(spans),
        "not_clearly_covered_span_classes": len(unclear),
        "nonadjacent_drill_events": nonadjacent_drills,
        "dielectric_gaps_below_published_minimum": dielectric_violations,
        "evaluation_required_above_30_layers": (
            int(manifest["layer_count"]) > 30
        ),
        "blocking_questions": [
            "Provide a manufacturable material stack and finished thickness.",
            "Map each requested blind/buried span to a lamination/drill step.",
            "Confirm which deep mechanical spans are accepted or must change.",
            "Quote the exact span schedule; do not substitute microvias.",
        ],
    }
    (output_dir / "dfm-review.json").write_text(
        json.dumps(review, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    readme = [
        "# PCBWay DFM review packet",
        "",
        "**Status: prepared for review; not fabricator-approved.**",
        "",
        f"- Routed KiCad board: `{board}`",
        f"- Copper layers: {review['layer_count']}",
        f"- Provisional thickness: {review['board_thickness_mm']} mm",
        f"- Drill events: {total_drills}",
        f"- Span classes: {len(spans)}",
        f"- Span classes not clearly covered: {len(unclear)}",
        f"- Nonadjacent drill events: {nonadjacent_drills}",
        f"- Dielectric gaps below published 2 mil minimum: "
        f"{dielectric_violations}",
        "",
        "## Questions for PCBWay",
        "",
    ]
    readme.extend(f"- {item}" for item in review["blocking_questions"])
    readme.extend([
        "",
        "## Published PCBWay references used",
        "",
    ])
    readme.extend(
        f"- [{source['title']}]({source['url']}): {source['used_for']}."
        for source in PCBWAY_SOURCES
    )
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8", newline="\n"
    )
    archive = shutil.make_archive(
        str(output_dir),
        "zip",
        root_dir=output_dir.parent,
        base_dir=output_dir.name,
    )
    return {
        **review,
        "packet_directory": str(output_dir),
        "packet_archive": archive,
        "readme": str(output_dir / "README.md"),
        "stackup_csv": str(stackup_csv),
        "span_review_csv": str(span_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fabrication_manifest", type=Path)
    parser.add_argument("drc_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_packet(
        args.fabrication_manifest,
        args.drc_summary,
        args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
