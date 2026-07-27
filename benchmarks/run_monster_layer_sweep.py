"""Supervise full mechanical monster routes and export the lowest success.

This script can attach to an already-running initial candidate by monitoring
its progress journal. Subsequent layer candidates run to their configured
terminal condition; plateaus never trigger an early stop.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2), encoding="utf-8", newline="\n"
    )
    temporary.replace(path)


def _wait_for_terminal(
    progress_path: Path,
    state: Dict[str, Any],
    state_path: Path,
) -> Dict[str, Any]:
    while True:
        journal = _read_json(progress_path)
        state["active_progress"] = str(progress_path)
        state["active_status"] = journal.get("status")
        state["active_iteration"] = len(journal.get("iterations", []))
        state["updated"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(state_path, state)
        if journal.get("status") in {"complete", "incomplete", "failed"}:
            return journal
        time.sleep(30)


def _latest_progress(results_dir: Path, layer_count: int) -> Optional[Path]:
    candidates = sorted(
        results_dir.glob(
            f"Backplane-{layer_count}L-{layer_count}L-"
            "*HDI-pcbway_mechanical-*-progress.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run_candidate(
    repo_root: Path,
    results_dir: Path,
    layer_count: int,
    max_iterations: int,
) -> Tuple[Path, Dict[str, Any]]:
    previous = _latest_progress(results_dir, layer_count)
    environment = os.environ.copy()
    environment.pop("ORTHO_DIAGNOSTIC_NET_LIMIT", None)
    environment.update({
        "ORTHO_DIAGNOSTIC_LAYER_LIMIT": str(layer_count),
        "ORTHO_DIAGNOSTIC_OWNER_PENALTY": "10",
        "ORTHO_DIAGNOSTIC_PATH_NODE_PENALTY": "10",
        "ORTHO_DIAGNOSTIC_MAX_ITERATIONS": str(max_iterations),
        "ORTHO_FAB_PROFILE": "pcbway_advanced_hdi",
        "ORTHO_DIRECTION_MODE": "guided",
        "ORTHO_WRONG_WAY_MULTIPLIER": "4",
        "ORTHO_LAYER_DEPTH_BIAS": "0.02",
        "ORTHO_HDI_STACK": "pcbway_mechanical",
        "ORTHO_GRID_PITCH": "0.4",
    })
    stdout_path = (
        repo_root / "benchmarks" / "results"
        / f"monster-full-{layer_count}L-mechanical-sweep.stdout.log"
    )
    stderr_path = (
        repo_root / "benchmarks" / "results"
        / f"monster-full-{layer_count}L-mechanical-sweep.stderr.log"
    )
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        return_code = subprocess.run(
            [
                sys.executable,
                str(
                    repo_root / "benchmarks" / "results"
                    / "monster_full_route.py"
                ),
            ],
            cwd=repo_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        ).returncode
    progress = _latest_progress(results_dir, layer_count)
    if (
        progress is None
        or progress == previous
        or progress.stat().st_mtime < started
    ):
        raise RuntimeError(
            f"{layer_count}L route returned {return_code} without a journal"
        )
    return progress, _read_json(progress)


def _refresh_comparison(repo_root: Path, results_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "summarize_route_progress.py"),
            str(results_dir),
            str(results_dir),
        ],
        cwd=repo_root,
        check=True,
    )


def _artifact_from_journal(journal: Dict[str, Any], key: str) -> Path:
    direct = journal.get(key)
    if direct:
        return Path(direct)
    metrics_path = journal.get("metrics")
    if not metrics_path:
        raise RuntimeError(f"terminal journal has no {key} or metrics path")
    metrics = _read_json(Path(metrics_path))
    return Path(metrics["artifacts"][key])


def _export_and_drc(
    repo_root: Path,
    results_dir: Path,
    source_board: Path,
    layer_count: int,
    journal: Dict[str, Any],
) -> Dict[str, Any]:
    geometry = _artifact_from_journal(journal, "geometry")
    stem = f"Backplane-OrthoRoute-{layer_count}L-FULL-MECHANICAL"
    board = results_dir / f"{stem}.kicad_pcb"
    project = results_dir / f"{stem}.kicad_pro"
    drc = results_dir / f"{stem}-drc.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "export_kicad_board.py"),
            str(geometry),
            str(source_board),
            str(board),
            "--layers",
            str(layer_count),
            "--thickness",
            "1.6",
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            "kicad-cli",
            "pcb",
            "drc",
            "--format",
            "json",
            "--output",
            str(drc),
            str(board),
        ],
        cwd=repo_root,
        check=True,
    )
    report = _read_json(drc)
    return {
        "board": str(board),
        "project": str(project),
        "drc": str(drc),
        "drc_errors": sum(
            item.get("severity") == "error"
            for item in report.get("violations", [])
        ),
        "drc_warnings": sum(
            item.get("severity") == "warning"
            for item in report.get("violations", [])
        ),
        "unconnected_items": len(report.get("unconnected_items", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_progress", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("source_board", type=Path)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--max-layers", type=int, default=20)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    state_path = (
        repo_root / "benchmarks" / "results"
        / "monster-layer-sweep-state.json"
    )
    state: Dict[str, Any] = {
        "status": "monitoring_initial",
        "started": datetime.now().isoformat(timespec="seconds"),
        "runs": [],
    }
    _atomic_json(state_path, state)

    initial = _wait_for_terminal(args.initial_progress, state, state_path)
    initial_layers = int(
        re.search(r"Backplane-(\d+)L", initial["run_name"]).group(1)
    )
    initial_complete = bool(
        initial.get("completion", {}).get("complete", False)
    )
    state["runs"].append({
        "layers": initial_layers,
        "progress": str(args.initial_progress),
        "status": initial.get("status"),
        "complete": initial_complete,
    })
    selected: Optional[Tuple[int, Dict[str, Any]]] = (
        (initial_layers, initial) if initial_complete else None
    )

    if initial_complete and initial_layers > 14:
        candidates = [14]
    else:
        candidates = list(
            range(initial_layers + 2, args.max_layers + 1, 2)
        )
    for layer_count in candidates:
        state["status"] = f"routing_{layer_count}L"
        _atomic_json(state_path, state)
        progress, journal = _run_candidate(
            repo_root,
            args.results_dir,
            layer_count,
            args.max_iterations,
        )
        complete = bool(
            journal.get("completion", {}).get("complete", False)
        )
        state["runs"].append({
            "layers": layer_count,
            "progress": str(progress),
            "status": journal.get("status"),
            "complete": complete,
        })
        _refresh_comparison(repo_root, args.results_dir)
        if complete:
            if selected is None or layer_count < selected[0]:
                selected = (layer_count, journal)
            break

    if selected is None:
        state["status"] = "no_complete_candidate"
        _atomic_json(state_path, state)
        raise SystemExit("no layer candidate completed")
    state["status"] = "exporting"
    _atomic_json(state_path, state)
    state["selected_layers"] = selected[0]
    state["deliverable"] = _export_and_drc(
        repo_root,
        args.results_dir,
        args.source_board,
        selected[0],
        selected[1],
    )
    _refresh_comparison(repo_root, args.results_dir)
    state["status"] = "complete"
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(state_path, state)


if __name__ == "__main__":
    main()
