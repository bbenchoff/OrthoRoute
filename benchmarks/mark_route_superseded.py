"""Atomically mark a preserved route journal as intentionally superseded."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def mark_superseded(
    progress_path: Path,
    reason: str,
    *,
    minimum_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    """Preserve every snapshot and make an intentional stop terminal."""
    journal = json.loads(progress_path.read_text(encoding="utf-8"))
    previous_status = str(journal.get("status", ""))
    if previous_status not in {"starting", "routing"}:
        raise ValueError(
            f"refusing to supersede terminal status {previous_status!r}"
        )
    iteration = len(journal.get("iterations", []))
    if minimum_iteration is not None and iteration < minimum_iteration:
        raise ValueError(
            f"journal has iteration {iteration}, expected at least "
            f"{minimum_iteration}"
        )
    reason = reason.strip()
    if not reason:
        raise ValueError("supersession reason must not be empty")

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    journal["status"] = "failed"
    journal["updated"] = timestamp
    journal["previous_status"] = previous_status
    journal["termination_reason"] = reason
    journal["superseded"] = {
        "reason": reason,
        "iteration": iteration,
        "timestamp": timestamp,
        "snapshots_preserved": True,
    }
    journal["error"] = f"superseded: {reason}"
    _atomic_json(progress_path, journal)
    return journal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress_path", type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--minimum-iteration", type=int)
    args = parser.parse_args()
    journal = mark_superseded(
        args.progress_path,
        args.reason,
        minimum_iteration=args.minimum_iteration,
    )
    print(json.dumps({
        "progress": str(args.progress_path),
        "status": journal["status"],
        "iteration": len(journal.get("iterations", [])),
        "termination_reason": journal["termination_reason"],
    }, indent=2))


if __name__ == "__main__":
    main()
