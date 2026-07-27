"""Export OrthoRoute geometry into a reduced-layer KiCad board."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orthoroute.infrastructure.kicad.text_board_exporter import (
    export_geometry_to_board,
    export_project_for_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--thickness", type=float, default=1.6)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--no-consolidate-mechanical-vias",
        action="store_true",
        help="preserve adjacent CNC via objects instead of merging stacks",
    )
    args = parser.parse_args()
    result = export_geometry_to_board(
        args.geometry,
        args.source,
        args.output,
        layer_count=args.layers,
        thickness_mm=args.thickness,
        limit=args.limit,
        consolidate_mechanical_vias=(
            not args.no_consolidate_mechanical_vias
        ),
    )
    source_project = args.source.with_suffix(".kicad_pro")
    if source_project.exists():
        result.update(export_project_for_geometry(
            args.geometry,
            source_project,
            args.output.with_suffix(".kicad_pro"),
        ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
