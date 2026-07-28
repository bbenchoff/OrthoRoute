"""Supervise full mechanical routes and export the lowest practical result.

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
from typing import Any, Dict, List, Optional, Tuple


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
    refresh_comparison=None,
) -> Dict[str, Any]:
    last_refresh_iteration = -10
    while True:
        journal = _read_json(progress_path)
        state["active_progress"] = str(progress_path)
        state["active_status"] = journal.get("status")
        state["active_iteration"] = len(journal.get("iterations", []))
        state["updated"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(state_path, state)
        if (
            refresh_comparison is not None
            and state["active_iteration"] >= last_refresh_iteration + 10
        ):
            refresh_comparison()
            last_refresh_iteration = state["active_iteration"]
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
    source_board: Path,
    layer_count: int,
    max_iterations: int,
    layer_depth_bias: float,
    state: Dict[str, Any],
    state_path: Path,
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
        "ORTHO_LAYER_DEPTH_BIAS": str(layer_depth_bias),
        "ORTHO_HDI_STACK": "pcbway_mechanical",
        "ORTHO_GRID_PITCH": "0.4",
        "ORTHO_SOURCE_BOARD": str(source_board.resolve()),
        "ORTHO_OUTPUT_DIR": str(results_dir.resolve()),
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
    return_code = None
    active_progress = None
    last_refresh_iteration = -10
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
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
        )
        while process.poll() is None:
            candidate = _latest_progress(results_dir, layer_count)
            if (
                candidate is not None
                and candidate != previous
                and candidate.stat().st_mtime >= started
            ):
                active_progress = candidate
                journal = _read_json(candidate)
                iteration = len(journal.get("iterations", []))
                state["active_progress"] = str(candidate)
                state["active_status"] = journal.get("status")
                state["active_iteration"] = iteration
                state["updated"] = datetime.now().isoformat(
                    timespec="seconds"
                )
                _atomic_json(state_path, state)
                if iteration >= last_refresh_iteration + 10:
                    _refresh_comparison(repo_root, results_dir)
                    last_refresh_iteration = iteration
            time.sleep(30)
        return_code = process.wait()
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


def _remaining_candidates(
    initial_layers: int,
    selected: Optional[Tuple[int, Dict[str, Any]]],
    max_layers: int,
    candidate_layers: Optional[List[int]] = None,
) -> List[int]:
    """Choose lower qualification or upward capacity candidates."""
    if selected is not None:
        return [selected[0] - 2] if selected[0] > 14 else []
    if candidate_layers is not None:
        return [
            layers for layers in candidate_layers
            if initial_layers < layers <= max_layers
        ]
    return list(range(initial_layers + 2, max_layers + 1, 2))


def _parse_candidate_layers(
    value: Optional[str],
    *,
    initial_layers: int,
    max_layers: int,
) -> Optional[List[int]]:
    """Validate an optional strictly increasing coarse layer ladder."""
    if value is None:
        return None
    try:
        layers = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "candidate layers must be comma-separated integers"
        ) from error
    if not layers or any(layer % 2 for layer in layers):
        raise ValueError("candidate layers must be non-empty and even")
    if layers != sorted(set(layers)):
        raise ValueError(
            "candidate layers must be unique and strictly increasing"
        )
    if any(
        layer <= initial_layers or layer > max_layers
        for layer in layers
    ):
        raise ValueError(
            "candidate layers must be above the initial layer count "
            "and at or below max layers"
        )
    return layers


def _backfill_candidates(
    failed_layers: int,
    accepted_layers: int,
) -> List[int]:
    """Return untested even layers between a failure and coarse success."""
    return list(range(failed_layers + 2, accepted_layers, 2))


def _artifact_from_journal(journal: Dict[str, Any], key: str) -> Path:
    direct = journal.get(key)
    if direct:
        return Path(direct)
    metrics_path = journal.get("metrics")
    if not metrics_path:
        raise RuntimeError(f"terminal journal has no {key} or metrics path")
    metrics = _read_json(Path(metrics_path))
    return Path(metrics["artifacts"][key])


def _drc_counts(report: Dict[str, Any]) -> Dict[str, int]:
    """Count every KiCad item that still requires electrical cleanup."""
    rule_errors = sum(
        item.get("severity") == "error"
        for item in report.get("violations", [])
    )
    warnings = sum(
        item.get("severity") == "warning"
        for item in report.get("violations", [])
    )
    unconnected = len(report.get("unconnected_items", []))
    return {
        "drc_errors": rule_errors,
        "drc_warnings": warnings,
        "unconnected_items": unconnected,
        # KiCad stores unrouted connections separately from rule errors, but
        # both require attention before this deliverable can be called routed.
        "reported_errors": rule_errors + unconnected,
    }


def _fabrication_manifest_paths(board: Path) -> Dict[str, str]:
    stem = board.with_name(board.stem + "-fabrication")
    return {
        "fabrication_manifest_json": str(stem.with_suffix(".json")),
        "fabrication_manifest_csv": str(stem.with_suffix(".csv")),
        "fabrication_manifest_markdown": str(stem.with_suffix(".md")),
    }


def _export_and_drc(
    repo_root: Path,
    results_dir: Path,
    source_board: Path,
    layer_count: int,
    journal: Dict[str, Any],
    *,
    label: str = "FULL",
) -> Dict[str, Any]:
    geometry = _artifact_from_journal(journal, "geometry")
    stem = f"Backplane-OrthoRoute-{layer_count}L-{label}-MECHANICAL"
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
    drc_summary_stem = drc.with_name(drc.stem + "-summary")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "summarize_kicad_drc.py"),
            str(drc),
            "--output-stem",
            str(drc_summary_stem),
        ],
        cwd=repo_root,
        check=True,
    )
    report = _read_json(drc)
    return {
        "board": str(board),
        "project": str(project),
        "drc": str(drc),
        "drc_summary_markdown": str(
            drc_summary_stem.with_suffix(".md")
        ),
        "drc_summary_csv": str(drc_summary_stem.with_suffix(".csv")),
        **_fabrication_manifest_paths(board),
        **_drc_counts(report),
    }


def _qualification_label(journal: Dict[str, Any]) -> str:
    match = re.search(
        r"(\d{8}_\d{6})$",
        str(journal.get("run_name", "")),
    )
    return "QUAL-" + (match.group(1) if match else "CURRENT")


def _qualify_candidate(
    repo_root: Path,
    results_dir: Path,
    source_board: Path,
    layer_count: int,
    journal: Dict[str, Any],
    drc_error_target: int,
) -> Dict[str, Any]:
    """Use exported KiCad DRC, not zero internal conflicts, as acceptance."""
    strict_complete = bool(
        journal.get("completion", {}).get("complete", False)
    )
    result: Dict[str, Any] = {
        "strict_complete": strict_complete,
        "accepted": False,
    }
    if journal.get("status") == "failed":
        result["reason"] = "route_failed_without_geometry"
        return result
    try:
        deliverable = _export_and_drc(
            repo_root,
            results_dir,
            source_board,
            layer_count,
            journal,
            label=_qualification_label(journal),
        )
    except Exception as exc:
        result["reason"] = (
            f"qualification_export_failed: {type(exc).__name__}: {exc}"
        )
        return result
    deliverable["drc_target_met"] = (
        deliverable["reported_errors"] < drc_error_target
    )
    result["deliverable"] = deliverable
    result["accepted"] = deliverable["drc_target_met"]
    result["reason"] = (
        "kicad_drc_below_target"
        if result["accepted"]
        else "kicad_drc_above_target"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_progress", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("source_board", type=Path)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--max-layers", type=int, default=20)
    parser.add_argument(
        "--layer-depth-bias",
        type=float,
        default=0.02,
        help=(
            "Monotonic cost added per deeper layer; use zero when testing "
            "whether depth packing constrains a minimum-layer candidate"
        ),
    )
    parser.add_argument(
        "--candidate-layers",
        help=(
            "Comma-separated coarse expansion ladder; after its first "
            "success, untested even layers above the last failure are "
            "backfilled to preserve the lowest-layer result"
        ),
    )
    parser.add_argument(
        "--retry-initial",
        action="store_true",
        help=(
            "rerun an incomplete attached candidate with current code "
            "before adding layers"
        ),
    )
    parser.add_argument(
        "--drc-error-target",
        type=int,
        default=100,
        help="Require rule errors plus unconnected items below this count",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    state_path = (
        repo_root / "benchmarks" / "results"
        / "monster-layer-sweep-state.json"
    )
    state: Dict[str, Any] = {
        "status": "monitoring_initial",
        "started": datetime.now().isoformat(timespec="seconds"),
        "candidate_max_iterations": args.max_iterations,
        "max_layers": args.max_layers,
        "candidate_layers": args.candidate_layers,
        "layer_depth_bias": args.layer_depth_bias,
        "retry_initial": args.retry_initial,
        "drc_error_target": args.drc_error_target,
        "runs": [],
    }
    _atomic_json(state_path, state)

    initial = _wait_for_terminal(
        args.initial_progress,
        state,
        state_path,
        refresh_comparison=lambda: _refresh_comparison(
            repo_root, args.results_dir
        ),
    )
    initial_layers = int(
        re.search(r"Backplane-(\d+)L", initial["run_name"]).group(1)
    )
    candidate_layers = _parse_candidate_layers(
        args.candidate_layers,
        initial_layers=initial_layers,
        max_layers=args.max_layers,
    )
    state["candidate_layers"] = candidate_layers
    _atomic_json(state_path, state)
    initial_qualification = _qualify_candidate(
        repo_root,
        args.results_dir,
        args.source_board,
        initial_layers,
        initial,
        args.drc_error_target,
    )
    state["runs"].append({
        "layers": initial_layers,
        "progress": str(args.initial_progress),
        "status": initial.get("status"),
        "complete": initial_qualification["strict_complete"],
        "accepted": initial_qualification["accepted"],
        "qualification": initial_qualification,
        "reason": "attached_baseline",
    })
    selected: Optional[Tuple[int, Dict[str, Any]]] = (
        (initial_layers, initial)
        if initial_qualification["accepted"] else None
    )

    if selected is None and args.retry_initial:
        state["status"] = f"retrying_{initial_layers}L"
        _atomic_json(state_path, state)
        progress, retry = _run_candidate(
            repo_root,
            args.results_dir,
            args.source_board,
            initial_layers,
            args.max_iterations,
            args.layer_depth_bias,
            state,
            state_path,
        )
        retry_qualification = _qualify_candidate(
            repo_root,
            args.results_dir,
            args.source_board,
            initial_layers,
            retry,
            args.drc_error_target,
        )
        state["runs"].append({
            "layers": initial_layers,
            "progress": str(progress),
            "status": retry.get("status"),
            "complete": retry_qualification["strict_complete"],
            "accepted": retry_qualification["accepted"],
            "qualification": retry_qualification,
            "reason": "current_code_retry",
        })
        _refresh_comparison(repo_root, args.results_dir)
        if retry_qualification["accepted"]:
            selected = (initial_layers, retry)

    candidates = _remaining_candidates(
        initial_layers,
        selected,
        args.max_layers,
        candidate_layers=candidate_layers,
    )
    last_failed_layers = initial_layers
    coarse_success_layers: Optional[int] = None
    for layer_count in candidates:
        state["status"] = f"routing_{layer_count}L"
        _atomic_json(state_path, state)
        progress, journal = _run_candidate(
            repo_root,
            args.results_dir,
            args.source_board,
            layer_count,
            args.max_iterations,
            args.layer_depth_bias,
            state,
            state_path,
        )
        qualification = _qualify_candidate(
            repo_root,
            args.results_dir,
            args.source_board,
            layer_count,
            journal,
            args.drc_error_target,
        )
        state["runs"].append({
            "layers": layer_count,
            "progress": str(progress),
            "status": journal.get("status"),
            "complete": qualification["strict_complete"],
            "accepted": qualification["accepted"],
            "qualification": qualification,
            "reason": (
                "lower_layer_qualification"
                if selected is not None
                else "capacity_expansion"
            ),
        })
        _refresh_comparison(repo_root, args.results_dir)
        if qualification["accepted"]:
            if selected is None or layer_count < selected[0]:
                selected = (layer_count, journal)
            coarse_success_layers = layer_count
            break
        last_failed_layers = layer_count

    if (
        selected is not None
        and coarse_success_layers is not None
        and candidate_layers is not None
    ):
        for layer_count in _backfill_candidates(
            last_failed_layers,
            coarse_success_layers,
        ):
            state["status"] = f"backfilling_{layer_count}L"
            _atomic_json(state_path, state)
            progress, journal = _run_candidate(
                repo_root,
                args.results_dir,
                args.source_board,
                layer_count,
                args.max_iterations,
                args.layer_depth_bias,
                state,
                state_path,
            )
            qualification = _qualify_candidate(
                repo_root,
                args.results_dir,
                args.source_board,
                layer_count,
                journal,
                args.drc_error_target,
            )
            state["runs"].append({
                "layers": layer_count,
                "progress": str(progress),
                "status": journal.get("status"),
                "complete": qualification["strict_complete"],
                "accepted": qualification["accepted"],
                "qualification": qualification,
                "reason": "minimum_layer_backfill",
            })
            _refresh_comparison(repo_root, args.results_dir)
            if qualification["accepted"]:
                selected = (layer_count, journal)
                break

    if selected is None:
        state["status"] = "no_practical_candidate"
        _atomic_json(state_path, state)
        raise SystemExit("no layer candidate met the KiCad DRC target")
    state["status"] = "exporting"
    _atomic_json(state_path, state)
    state["selected_layers"] = selected[0]
    deliverable = _export_and_drc(
        repo_root,
        args.results_dir,
        args.source_board,
        selected[0],
        selected[1],
    )
    deliverable["drc_target_met"] = (
        deliverable["reported_errors"] < args.drc_error_target
    )
    state["deliverable"] = deliverable
    _refresh_comparison(repo_root, args.results_dir)
    state["status"] = (
        "complete"
        if deliverable["drc_target_met"]
        else "drc_target_not_met"
    )
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(state_path, state)


if __name__ == "__main__":
    main()
