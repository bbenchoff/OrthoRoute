"""Routing metrics collection.

Turns a routed PathFinderRouter into a JSON-serializable dict answering
the questions that matter for the speed / fewer-layers work: which layers
actually carry copper, how much wirelength and how many vias, did every
net route, how many iterations did convergence take, and how long each
phase cost. Run-over-run comparison is the whole point - keep keys stable.
"""

import subprocess
from typing import Dict, Optional


def collect_route_metrics(pf, board, timings: Optional[Dict[str, float]] = None) -> Dict:
    """Collect metrics from a routed PathFinderRouter instance.

    Args:
        pf: Router after route_multiple_nets (and ideally emit_geometry).
        board: The routed domain Board.
        timings: Optional {phase_name: seconds} measured by the caller.

    Returns:
        JSON-serializable metrics dict.
    """
    plane = pf.lattice.x_steps * pf.lattice.y_steps
    pitch = pf.lattice.pitch
    preferred_axis = getattr(
        pf.lattice,
        "get_legal_axis",
        lambda layer: "h" if layer % 2 == 1 else "v",
    )

    lateral_steps_per_layer: Dict[int, int] = {}
    preferred_steps_per_layer: Dict[int, int] = {}
    wrong_way_steps_per_layer: Dict[int, int] = {}
    via_transitions = 0
    via_layer_steps = 0
    routed_path_ids = set()
    trivial_path_ids = set()

    for net_id, path in pf.net_paths.items():
        if len(path) == 1:
            trivial_path_ids.add(str(net_id))
            continue
        if len(path) < 2:
            continue
        routed_path_ids.add(str(net_id))
        in_vertical_run = False
        vertical_xy = None
        vertical_direction = 0
        for a, b in zip(path, path[1:]):
            za, zb = a // plane, b // plane
            if za == zb:
                lateral_steps_per_layer[za] = lateral_steps_per_layer.get(za, 0) + 1
                xa = (a % plane) % pf.lattice.x_steps
                xb = (b % plane) % pf.lattice.x_steps
                axis = "h" if xa != xb else "v"
                counter = (
                    preferred_steps_per_layer
                    if axis == preferred_axis(za)
                    else wrong_way_steps_per_layer
                )
                counter[za] = counter.get(za, 0) + 1
                in_vertical_run = False
                vertical_xy = None
                vertical_direction = 0
            else:
                via_layer_steps += abs(zb - za)
                xy = a % plane
                direction = 1 if zb > za else -1
                if (
                    not in_vertical_run
                    or xy != vertical_xy
                    or direction != vertical_direction
                ):
                    via_transitions += 1
                in_vertical_run = True
                vertical_xy = xy
                vertical_direction = direction

    raw_nets = list(board.nets)
    routable_ids = {
        str(getattr(net, "name", None) or getattr(net, "id", ""))
        for net in raw_nets
        if len(getattr(net, "pads", ())) >= 2
    }
    completed_ids = (routed_path_ids | trivial_path_ids) & routable_ids
    unrouted_ids = sorted(routable_ids - completed_ids)
    excluded_ids = {
        str(net_id) for net_id in getattr(pf, "_excluded_nets", ())
    }
    total_lateral = sum(lateral_steps_per_layer.values())
    overuse_total, overuse_count = pf.accounting.compute_overuse(pf)
    barrel_conflicts = int(
        getattr(pf, "_last_barrel_conflict_count", 0)
    )
    exact_barrel_conflicts = int(
        getattr(pf, "_last_exact_barrel_conflict_count", 0)
    )
    escape_conflicts = int(
        getattr(pf, "_last_escape_conflict_count", 0)
    )
    portal_grid_conflicts = int(
        getattr(pf, "_last_portal_grid_conflict_count", 0)
    )

    return {
        "board": {
            "name": board.name,
            "layer_count": board.layer_count,
            "nets": len(raw_nets),
            "routable_nets": len(routable_ids),
            "singleton_nets": len(raw_nets) - len(routable_ids),
            "pads": sum(len(getattr(n, "pads", [])) for n in raw_nets),
        },
        "lattice": {
            "x_steps": pf.lattice.x_steps,
            "y_steps": pf.lattice.y_steps,
            "layers": pf.lattice.layers,
            "nodes": pf.lattice.num_nodes,
            "pitch_mm": pitch,
        },
        "completion": {
            "routed_nets": len(routed_path_ids & routable_ids),
            "trivial_nets": len(trivial_path_ids & routable_ids),
            "completed_nets": len(completed_ids),
            "total_nets": len(routable_ids),
            "excluded_nets": len(excluded_ids),
            "excluded_net_ids": sorted(excluded_ids),
            "unrouted_net_ids": unrouted_ids,
            "complete": (
                len(completed_ids) == len(routable_ids)
                and not excluded_ids
                and int(overuse_total) == 0
                and barrel_conflicts == 0
            ),
        },
        "convergence": {
            "iterations": getattr(pf, "iteration", None),
            "overuse_total": int(overuse_total),
            "overuse_count": int(overuse_count),
            "barrel_conflicts": barrel_conflicts,
            "exact_barrel_conflicts": exact_barrel_conflicts,
            "escape_conflicts": escape_conflicts,
            "portal_grid_conflicts": portal_grid_conflicts,
        },
        "copper": {
            "wirelength_mm": round(total_lateral * pitch, 3),
            "via_transitions": via_transitions,
            "via_layer_steps": via_layer_steps,
            "lateral_steps_per_layer": {
                str(z): n for z, n in sorted(lateral_steps_per_layer.items())
            },
            "preferred_steps_per_layer": {
                str(z): n
                for z, n in sorted(preferred_steps_per_layer.items())
            },
            "wrong_way_steps_per_layer": {
                str(z): n
                for z, n in sorted(wrong_way_steps_per_layer.items())
            },
            "wrong_way_steps": sum(wrong_way_steps_per_layer.values()),
            "layers_used": sorted(lateral_steps_per_layer.keys()),
            "layers_used_count": len(lateral_steps_per_layer),
        },
        "timings_s": {k: round(v, 3) for k, v in (timings or {}).items()},
        "git_sha": _git_sha(),
    }


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None
