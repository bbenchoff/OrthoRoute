import gzip
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cupy as cp


repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmarks.metrics import collect_route_metrics
import orthoroute.algorithms.manhattan.unified_pathfinder as pathfinder_module
from orthoroute.algorithms.manhattan.unified_pathfinder import (
    PathFinderConfig,
    UnifiedPathFinder,
)
from orthoroute.algorithms.manhattan.hdi_stack import (
    pcbway_elic_stack,
    pcbway_mechanical_stack,
)
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser


source_board_value = os.getenv("ORTHO_SOURCE_BOARD")
if not source_board_value:
    raise RuntimeError(
        "ORTHO_SOURCE_BOARD must name the KiCad board to route"
    )
source_board = Path(source_board_value).expanduser().resolve()
output_dir = Path(os.getenv(
    "ORTHO_OUTPUT_DIR",
    str(source_board.parent / "OrthoRoute-results"),
)).expanduser().resolve()
output_dir.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
diagnostic_max_iterations = os.getenv("ORTHO_DIAGNOSTIC_MAX_ITERATIONS")
diagnostic_net_limit = int(os.getenv("ORTHO_DIAGNOSTIC_NET_LIMIT", "0"))
diagnostic_layer_limit = int(os.getenv(
    "ORTHO_DIAGNOSTIC_LAYER_LIMIT", "0"
))
diagnostic_via_scale = os.getenv("ORTHO_DIAGNOSTIC_VIA_SCALE")
diagnostic_owner_penalty = os.getenv("ORTHO_DIAGNOSTIC_OWNER_PENALTY")
diagnostic_path_node_penalty = os.getenv(
    "ORTHO_DIAGNOSTIC_PATH_NODE_PENALTY"
)
fabrication_profile = os.getenv("ORTHO_FAB_PROFILE", "").strip()
direction_mode = os.getenv("ORTHO_DIRECTION_MODE", "strict").strip()
wrong_way_multiplier = float(os.getenv(
    "ORTHO_WRONG_WAY_MULTIPLIER", "4.0"
))
layer_directions = os.getenv("ORTHO_LAYER_DIRECTIONS", "").strip()
layer_depth_bias = float(os.getenv("ORTHO_LAYER_DEPTH_BIAS", "0.0"))
hdi_stack_mode = os.getenv("ORTHO_HDI_STACK", "").strip()
grid_pitch = float(os.getenv("ORTHO_GRID_PITCH", "0.4"))
run_tags = []
if diagnostic_net_limit:
    run_tags.append(f"{diagnostic_net_limit}N")
if diagnostic_layer_limit:
    run_tags.append(f"{diagnostic_layer_limit}L")
if diagnostic_via_scale is not None:
    run_tags.append(f"V{diagnostic_via_scale}")
if diagnostic_owner_penalty is not None:
    run_tags.append(f"O{diagnostic_owner_penalty}")
if diagnostic_path_node_penalty is not None:
    run_tags.append(f"P{diagnostic_path_node_penalty}")
if diagnostic_max_iterations is not None:
    run_tags.append(f"I{diagnostic_max_iterations}")
if fabrication_profile:
    run_tags.append(f"FAB-{fabrication_profile}")
if direction_mode != "strict":
    run_tags.append(
        "BIDIR" if direction_mode == "bidirectional"
        else f"GUIDED-{wrong_way_multiplier:g}"
    )
if layer_depth_bias:
    run_tags.append(f"DEPTH-{layer_depth_bias:g}")
if hdi_stack_mode:
    run_tags.append(f"HDI-{hdi_stack_mode}")
if grid_pitch != 0.4:
    run_tags.append(f"GRID-{grid_pitch:g}")
tag = f"-{'-'.join(run_tags)}" if run_tags else ""
run_layers = diagnostic_layer_limit or 32
run_name = f"Backplane-{run_layers}L{tag}-{stamp}"
progress_path = output_dir / f"{run_name}-progress.json"
metrics_path = output_dir / f"{run_name}-metrics.json"
geometry_path = output_dir / f"{run_name}-geometry.json"
paths_path = output_dir / f"{run_name}-paths.pkl.gz"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def memory_snapshot():
    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
    }


logging.basicConfig(
    level=getattr(
        logging,
        os.getenv("ORTHO_LOG_LEVEL", "WARNING").upper(),
        logging.WARNING,
    ),
    format="%(asctime)s %(levelname)s %(message)s",
)
os.environ["USE_GPU"] = "1"

started = time.perf_counter()
source_sha = hashlib.sha256(source_board.read_bytes()).hexdigest()
progress = {
    "status": "starting",
    "run_name": run_name,
    "source_board": str(source_board),
    "source_sha256": source_sha,
    "started": datetime.now().isoformat(timespec="seconds"),
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip(),
    "iterations": [],
    "fabrication_profile": fabrication_profile or None,
}
atomic_json(progress_path, progress)

timings = {}
memory = {"start": memory_snapshot()}
router = None
board = None
result = None

try:
    phase_started = time.perf_counter()
    board = KiCadFileParser().load_board(str(source_board))
    if board is None:
        raise RuntimeError("monster board parse failed")
    if fabrication_profile:
        if fabrication_profile != "pcbway_advanced_hdi":
            raise ValueError(
                "ORTHO_FAB_PROFILE must be pcbway_advanced_hdi"
            )
        # Conservative advanced-HDI envelope. PCBWay advertises local
        # 2/2 mil features, but 4/4 mil is the broadly manufacturable
        # baseline. The 0.15 mm mechanical blind/buried drill plus a
        # 3 mil annular ring gives the existing 0.3024 mm finished pad.
        board._design_rules = dict(board._design_rules)
        board._design_rules.update({
            "default_track_width": 0.1016,
            "default_clearance": 0.1016,
            "default_via_drill": 0.15,
            "default_via_diameter": 0.3024,
            "min_via_annular_width": 0.0762,
            "min_hole_to_hole": 0.2794,
            "min_hole_clearance": 0.1524,
        })
    if diagnostic_layer_limit:
        if not 4 <= diagnostic_layer_limit <= len(board.layers):
            raise ValueError(
                "ORTHO_DIAGNOSTIC_LAYER_LIMIT must be between 4 "
                f"and {len(board.layers)}"
            )
        # Retain F.Cu, the lowest requested internal layers, and B.Cu.
        # This measures whether negotiated packing can stay shallow while
        # preserving a valid outer-copper stack for geometry/export.
        board.layers = (
            board.layers[:diagnostic_layer_limit - 1]
            + [board.layers[-1]]
        )
        board.layer_count = diagnostic_layer_limit
        for position, layer in enumerate(board.layers):
            layer.stackup_position = position
        board._build_indexes()
    timings["parse_board"] = time.perf_counter() - phase_started
    memory["parse_board"] = memory_snapshot()

    config = PathFinderConfig()
    config.grid_pitch = grid_pitch
    if hdi_stack_mode:
        stack_factories = {
            "pcbway_elic": pcbway_elic_stack,
            "pcbway_mechanical": pcbway_mechanical_stack,
        }
        if hdi_stack_mode not in stack_factories:
            raise ValueError(
                "ORTHO_HDI_STACK must be pcbway_elic or "
                "pcbway_mechanical"
            )
        config.hdi_stack = stack_factories[hdi_stack_mode](
            board.layer_count
        )
    if direction_mode == "strict":
        config.wrong_way_cost_multiplier = float("inf")
    elif direction_mode == "bidirectional":
        config.wrong_way_cost_multiplier = 1.0
    elif direction_mode == "guided":
        config.wrong_way_cost_multiplier = wrong_way_multiplier
    else:
        raise ValueError(
            "ORTHO_DIRECTION_MODE must be strict, guided, or "
            "bidirectional"
        )
    if layer_directions:
        config.preferred_layer_directions = [
            axis.strip().lower()
            for axis in layer_directions.split(",")
        ]
    config.layer_depth_bias = layer_depth_bias
    if diagnostic_via_scale is not None:
        config.adjacent_via_step_scale = float(diagnostic_via_scale)
    if diagnostic_owner_penalty is not None:
        config.owner_penalty_base = float(diagnostic_owner_penalty)
    if diagnostic_path_node_penalty is not None:
        config.path_node_penalty_base = float(
            diagnostic_path_node_penalty
        )
    # Preserve the actual experiment knobs in the journal. Zero-valued
    # options intentionally disappear from the compact run name, and source
    # ancestry alone should not be required to reconstruct a long GPU run.
    progress["experiment_config"] = {
        "layer_count": int(board.layer_count),
        "net_limit": int(diagnostic_net_limit),
        "max_iterations_override": (
            None
            if diagnostic_max_iterations is None else
            int(diagnostic_max_iterations)
        ),
        "grid_pitch": float(config.grid_pitch),
        "fabrication_profile": fabrication_profile or None,
        "hdi_stack": hdi_stack_mode or None,
        "direction_mode": direction_mode,
        "wrong_way_multiplier": float(wrong_way_multiplier),
        "preferred_layer_directions": (
            list(config.preferred_layer_directions)
            if config.preferred_layer_directions is not None else
            None
        ),
        "layer_depth_bias": float(config.layer_depth_bias),
        "owner_penalty_base": float(config.owner_penalty_base),
        "path_node_penalty_base": float(
            config.path_node_penalty_base
        ),
        "ordinary_hotset_cap": int(config.hotset_cap),
        "slow_progress_hotset_cap": int(
            config.slow_progress_hotset_cap
        ),
        "slow_progress_hotset_cap_max": int(
            config.slow_progress_hotset_cap_max
        ),
        "slow_progress_pressure_after": int(
            config.slow_progress_pressure_after
        ),
        "slow_progress_pres_fac_max": float(
            config.slow_progress_pres_fac_max
        ),
        "slow_progress_window": int(config.slow_progress_window),
        "slow_progress_min_fraction": float(
            config.slow_progress_min_fraction
        ),
    }
    atomic_json(progress_path, progress)
    if diagnostic_max_iterations is not None:
        original_apply_derived_parameters = (
            pathfinder_module.apply_derived_parameters
        )

        def apply_diagnostic_iteration_limit(config, derived):
            original_apply_derived_parameters(config, derived)
            config.max_iterations = int(diagnostic_max_iterations)

        pathfinder_module.apply_derived_parameters = (
            apply_diagnostic_iteration_limit
        )
    router = UnifiedPathFinder(config=config, use_gpu=True)

    route_nets = [
        net for net in board.nets
        if len(getattr(net, "pads", ())) >= 2
    ]
    if diagnostic_net_limit:
        route_nets = route_nets[:diagnostic_net_limit]

    phase_started = time.perf_counter()
    router.initialize_graph(board)
    cp.cuda.Stream.null.synchronize()
    timings["initialize_graph"] = time.perf_counter() - phase_started
    memory["initialize_graph"] = memory_snapshot()

    phase_started = time.perf_counter()
    router.precompute_all_pad_escapes(board, route_nets)
    cp.cuda.Stream.null.synchronize()
    timings["pad_escapes"] = time.perf_counter() - phase_started
    memory["pad_escapes"] = memory_snapshot()

    def progress_callback(iteration, total, message):
        routed = sum(bool(path) for path in router.net_paths.values())
        def overuse_sum(use, capacity):
            if use is None or capacity is None:
                return 0
            over = cp.maximum(use - capacity, 0)
            return int(cp.asnumpy(over.sum()))

        edge_over = cp.maximum(
            router.accounting.present
            - router.accounting.capacity,
            0,
        )
        canonical_edge_mask = cp.asarray(
            router._canonical_edge_resource_mask()
        )
        edge_overuse = int(cp.asnumpy(
            edge_over[canonical_edge_mask].sum()
        ))
        via_column_overuse = overuse_sum(
            getattr(router, "via_col_use", None),
            getattr(router, "via_col_cap", None),
        )
        via_segment_overuse = overuse_sum(
            getattr(router, "via_seg_use", None),
            getattr(router, "via_seg_cap", None),
        )
        (
            path_node_overuse,
            path_node_overuse_count,
        ) = router._compute_path_node_overuse()
        graph_overuse = (
            edge_overuse + via_column_overuse
            + via_segment_overuse
        )
        plane_size = (
            router.lattice.x_steps * router.lattice.y_steps
        )
        layers_used = sorted({
            int(node) // plane_size
            for path in router.net_paths.values()
            for node in path
            if 0 < int(node) // plane_size < router.lattice.layers - 1
        })
        progress["status"] = "routing"
        progress["updated"] = datetime.now().isoformat(timespec="seconds")
        progress["elapsed_seconds"] = round(
            time.perf_counter() - started, 3
        )
        progress["iteration"] = iteration
        progress["max_iterations"] = total
        progress["routed_nets"] = routed
        progress["excluded_nets"] = sorted(
            getattr(router, "_excluded_nets", ())
        )
        progress["pres_fac"] = float(
            getattr(router, "_pres_fac_now", 0.0)
        )
        progress["memory"] = memory_snapshot()
        physical_conflicts = int(getattr(
            router, "_last_barrel_conflict_count", 0
        ))
        capture_conflict_details = physical_conflicts <= 5_000
        escape_pair_options = []
        if capture_conflict_details:
            for first, second in sorted(getattr(
                router, "_escape_conflict_pairs", ()
            )):
                entry = {}
                for label, identity in (
                    ("first", first),
                    ("second", second),
                ):
                    net_name, pad_id = identity
                    candidates = router.portal_candidates.get(
                        pad_id, ()
                    )
                    conflicts = [
                        router._escape_candidate_conflicts(
                            net_name, pad_id, portal
                        )
                        for portal in candidates
                    ]
                    entry[f"{label}_net"] = net_name
                    entry[f"{label}_pad"] = pad_id
                    entry[f"{label}_candidates"] = len(candidates)
                    entry[f"{label}_clean_candidates"] = sum(
                        count == 0 for count in conflicts
                    )
                    entry[f"{label}_min_conflicts"] = (
                        min(conflicts) if conflicts else None
                    )
                escape_pair_options.append(entry)
        progress["iterations"].append({
            "iteration": iteration,
            "elapsed_seconds": progress["elapsed_seconds"],
            "routed_nets": routed,
            "excluded_nets": len(progress["excluded_nets"]),
            "pres_fac": progress["pres_fac"],
            "pres_fac_max": float(getattr(
                router, "_pres_fac_max_now", 0.0
            )),
            "pressure_work_scale": float(getattr(
                router, "_last_pressure_work_scale", 1.0
            )),
            "slow_progress_events": int(getattr(
                router, "_slow_progress_event_count", 0
            )),
            "slow_progress_fraction": getattr(
                router, "_last_slow_progress_fraction", None
            ),
            "hotset_rate_boost_until": int(getattr(
                router, "_hotset_rate_boost_until", 0
            )),
            "overuse_total": graph_overuse,
            "edge_overuse": edge_overuse,
            "via_column_overuse": via_column_overuse,
            "via_segment_overuse": via_segment_overuse,
            "path_node_overuse_total": path_node_overuse,
            "path_node_overuse_count": path_node_overuse_count,
            "path_node_layers": router._path_node_layer_metrics(),
            "negotiated_overuse_total": (
                graph_overuse + path_node_overuse
            ),
            "barrel_conflicts": physical_conflicts,
            "exact_barrel_conflicts": int(getattr(
                router, "_last_exact_barrel_conflict_count", 0
            )),
            "path_node_conflicts": int(getattr(
                router, "_last_path_node_conflict_count", 0
            )),
            "escape_conflicts": int(getattr(
                router, "_last_escape_conflict_count", 0
            )),
            "portal_grid_conflicts": int(getattr(
                router, "_last_portal_grid_conflict_count", 0
            )),
            "portal_grid_pairs": [
                {
                    "owner": identity[0],
                    "pad": identity[1],
                    "x_idx": int(identity[2]),
                    "y_idx": int(identity[3]),
                    "victim": victim,
                    "kind": kind,
                }
                for identity, victim, kind in sorted(getattr(
                    router, "_portal_grid_pairs", ()
                ))
            ] if capture_conflict_details else [],
            "conflict_details_truncated": (
                not capture_conflict_details
            ),
            "portal_cleanup_movable_nets": sorted(getattr(
                router, "_portal_cleanup_movable_nets", ()
            )),
            "stagnation_recovery_count": int(getattr(
                router, "stagnation_counter", 0
            )),
            "last_stagnation_victims": list(getattr(
                router, "_last_stagnation_victims", ()
            )),
            "hotset_size": int(getattr(
                router, "_last_hotset_size", 0
            )),
            "hotset_cap": int(getattr(
                router, "_last_hotset_cap", 0
            )),
            "hotset_offender_count": int(getattr(
                router, "_last_hotset_offender_count", 0
            )),
            "hotset_exploration_fraction": float(getattr(
                router,
                "_last_hotset_exploration_fraction",
                0.0,
            )),
            "hotset_conflict_aware": bool(getattr(
                router, "_last_hotset_conflict_aware", False
            )),
            "hotset_conflict_pair_count": int(getattr(
                router, "_last_hotset_conflict_pair_count", 0
            )),
            "hotset_conflict_pairs_covered": int(getattr(
                router, "_last_hotset_conflict_pairs_covered", 0
            )),
            "hotset_conflict_pair_coverage_fraction": float(getattr(
                router,
                "_last_hotset_conflict_pair_coverage_fraction",
                0.0,
            )),
            "via_pool_conflict_nets": len(getattr(
                router, "_via_pool_conflict_nets", ()
            )),
            "via_pool_keeper_resources": len(getattr(
                router, "_via_pool_keepers", {}
            )),
            "portal_cleanup_targets": [
                {
                    "net": identity[0],
                    "pad": identity[1],
                    "x_idx": int(portal.x_idx),
                    "y_idx": int(portal.y_idx),
                }
                for identity, portal in sorted(getattr(
                    router, "_portal_cleanup_target_portals", {}
                ).items())
            ],
            "escape_conflict_pairs": [
                {
                    "first_net": first[0],
                    "first_pad": first[1],
                    "second_net": second[0],
                    "second_pad": second[1],
                }
                for first, second in sorted(getattr(
                    router, "_escape_conflict_pairs", ()
                ))
            ] if capture_conflict_details else [],
            "escape_pair_options": (
                escape_pair_options
                if capture_conflict_details else []
            ),
            "exact_barrel_pairs": [
                {
                    "first_net": first,
                    "second_net": second,
                }
                for first, second in sorted(getattr(
                    router, "_exact_barrel_pairs", ()
                ))
            ] if capture_conflict_details else [],
            "barrel_details": (
                list(getattr(
                    router, "_last_exact_barrel_details", ()
                ))
                if capture_conflict_details else []
            ),
            "layers_used": layers_used,
            "memory": progress["memory"],
        })
        atomic_json(progress_path, progress)

    phase_started = time.perf_counter()
    result = router.route_multiple_nets(
        route_nets,
        progress_cb=progress_callback,
    )
    cp.cuda.Stream.null.synchronize()
    timings["route"] = time.perf_counter() - phase_started
    memory["route"] = memory_snapshot()

    phase_started = time.perf_counter()
    router.emit_geometry(board)
    timings["emit_geometry"] = time.perf_counter() - phase_started

    payload = router.get_geometry_payload()
    atomic_json(geometry_path, {
        "run_name": run_name,
        "source_board": str(source_board),
        "source_sha256": source_sha,
        "tracks": payload.tracks,
        "vias": payload.vias,
    })

    with gzip.open(paths_path, "wb", compresslevel=3) as stream:
        pickle.dump({
            "run_name": run_name,
            "source_board": str(source_board),
            "source_sha256": source_sha,
            "lattice": {
                "shape": (
                    router.lattice.x_steps,
                    router.lattice.y_steps,
                    router.lattice.layers,
                ),
                "pitch": router.lattice.pitch,
                "layer_directions": list(
                    router.lattice.layer_dir
                ),
            },
            "net_paths": router.net_paths,
            "net_portal_layers": router.net_portal_layers,
            "result": {
                key: value for key, value in result.items()
                if key != "paths"
            },
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)

    timings["total"] = time.perf_counter() - started
    metrics = collect_route_metrics(router, board, timings)
    selected_ids = {
        str(getattr(net, "name", None) or getattr(net, "id", ""))
        for net in route_nets
    }
    routed_ids = {
        str(net_id)
        for net_id, path in router.net_paths.items()
        if path and str(net_id) in selected_ids
    }
    excluded_ids = sorted(
        str(net_id)
        for net_id in getattr(router, "_excluded_nets", ())
        if str(net_id) in selected_ids
    )
    metrics["completion"] = {
        "routed_nets": len(routed_ids),
        "trivial_nets": 0,
        "completed_nets": len(routed_ids),
        "total_nets": len(selected_ids),
        "excluded_nets": len(excluded_ids),
        "excluded_net_ids": excluded_ids,
        "unrouted_net_ids": sorted(selected_ids - routed_ids),
        "complete": (
            routed_ids == selected_ids
            and not excluded_ids
            and metrics["convergence"]["overuse_total"] == 0
            and metrics["convergence"]["barrel_conflicts"] == 0
        ),
    }
    # The repository may advance while a multi-hour route is running.
    # Preserve the exact source revision captured at process start.
    metrics["git_sha"] = progress["git_sha"]
    metrics["params"] = {
        "backend": "gpu",
        "source_board": str(source_board),
        "source_sha256": source_sha,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "effective_max_iterations": config.max_iterations,
        "selected_routable_nets": len(route_nets),
        "layer_limit": diagnostic_layer_limit or board.layer_count,
        "adjacent_via_step_scale": config.adjacent_via_step_scale,
        "owner_penalty_base": config.owner_penalty_base,
        "path_node_penalty_base": config.path_node_penalty_base,
        "fabrication_profile": fabrication_profile or None,
        "direction_mode": direction_mode,
        "wrong_way_cost_multiplier": (
            config.wrong_way_cost_multiplier
        ),
        "preferred_layer_directions": list(
            router.lattice.layer_dir
        ),
        "configured_layer_directions": (
            config.preferred_layer_directions
        ),
        "layer_depth_bias": config.layer_depth_bias,
        "grid_pitch": config.grid_pitch,
        "hdi_stack": (
            config.hdi_stack.name
            if config.hdi_stack is not None else None
        ),
        "design_rules": dict(board._design_rules),
    }
    metrics["environment"] = {
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    metrics["memory"] = memory
    metrics["route_result"] = {
        key: value for key, value in result.items()
        if key != "paths"
    }
    metrics["artifacts"] = {
        "progress": str(progress_path),
        "geometry": str(geometry_path),
        "paths": str(paths_path),
    }
    atomic_json(metrics_path, metrics)

    progress["status"] = (
        "complete" if metrics["completion"]["complete"]
        and metrics["convergence"]["overuse_total"] == 0
        else "incomplete"
    )
    progress["updated"] = datetime.now().isoformat(timespec="seconds")
    progress["elapsed_seconds"] = round(timings["total"], 3)
    progress["metrics"] = str(metrics_path)
    progress["geometry"] = str(geometry_path)
    progress["paths"] = str(paths_path)
    progress["completion"] = metrics["completion"]
    progress["convergence"] = metrics["convergence"]
    atomic_json(progress_path, progress)

    print(json.dumps({
        "status": progress["status"],
        "metrics": str(metrics_path),
        "geometry": str(geometry_path),
        "paths": str(paths_path),
        "completion": metrics["completion"],
        "convergence": metrics["convergence"],
        "copper": metrics["copper"],
        "timings_s": metrics["timings_s"],
    }, indent=2), flush=True)
except Exception as exc:
    progress["status"] = "failed"
    progress["updated"] = datetime.now().isoformat(timespec="seconds")
    progress["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    progress["error"] = f"{type(exc).__name__}: {exc}"
    progress["traceback"] = traceback.format_exc()
    atomic_json(progress_path, progress)
    raise
