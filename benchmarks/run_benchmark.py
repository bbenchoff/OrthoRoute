"""CPU/GPU routing benchmark on a synthetic backplane.

Usage:
    python benchmarks/run_benchmark.py --cpu --connectors 2 --pins 16 --layers 4
    python benchmarks/run_benchmark.py --gpu --connectors 4 --pins 40 --layers 8

Writes a metrics JSON to benchmarks/results/ (gitignored) and prints a
one-line summary. Compare layers_used across --layers values to measure
how many layers a board actually needs.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.metrics import collect_route_metrics
from benchmarks.synthetic_boards import make_backplane


def run(args) -> dict:
    from orthoroute.algorithms.manhattan.unified_pathfinder import (
        PathFinderConfig, UnifiedPathFinder,
    )

    board = make_backplane(connectors=args.connectors, pins_per=args.pins,
                           layers=args.layers, pattern=args.pattern,
                           seed=args.seed)
    print(f"[BENCH] {board.name}: {len(board.nets)} nets, "
          f"{sum(len(n.pads) for n in board.nets)} pads")

    config = PathFinderConfig()
    if args.grid_pitch is not None:
        config.grid_pitch = args.grid_pitch
    if args.hdi_stack == "pcbway-elic":
        from orthoroute.algorithms.manhattan.hdi_stack import (
            pcbway_elic_stack,
        )
        config.hdi_stack = pcbway_elic_stack(args.layers)
    config.portal_x_snap_max = 0.75  # pads sit half a pitch off-grid (see conftest)
    if args.direction_mode == "strict":
        config.wrong_way_cost_multiplier = float("inf")
    elif args.direction_mode == "bidirectional":
        config.wrong_way_cost_multiplier = 1.0
    else:
        config.wrong_way_cost_multiplier = args.wrong_way_multiplier
    if args.layer_directions:
        config.preferred_layer_directions = [
            axis.strip().lower()
            for axis in args.layer_directions.split(",")
        ]
    config.layer_depth_bias = args.layer_depth_bias
    pf = UnifiedPathFinder(config=config, use_gpu=args.use_gpu)
    # Explicit benchmark selection wins over the router's legacy USE_GPU
    # environment override so CPU/GPU comparisons cannot silently swap backends.
    pf.config.use_gpu = args.use_gpu

    timings = {}
    memory = {}

    def sync_gpu():
        if pf.config.use_gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    def snapshot_gpu():
        if not pf.config.use_gpu:
            return None
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return {
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
        }

    def phase(name, fn):
        sync_gpu()
        t0 = time.perf_counter()
        result = fn()
        sync_gpu()
        timings[name] = time.perf_counter() - t0
        if pf.config.use_gpu:
            memory[name] = snapshot_gpu()
        print(f"[BENCH] {name}: {timings[name]:.2f}s")
        return result

    phase("initialize_graph", lambda: pf.initialize_graph(board))
    phase("map_all_pads", lambda: pf.map_all_pads(board))
    phase("pad_escapes", lambda: pf.precompute_all_pad_escapes(board))
    phase("prepare_runtime", lambda: pf.prepare_routing_runtime())
    phase("route", lambda: pf.route_multiple_nets(board.nets))
    phase("emit_geometry", lambda: pf.emit_geometry(board))
    timings["total"] = sum(timings.values())

    metrics = collect_route_metrics(pf, board, timings)
    metrics["params"] = {
        "connectors": args.connectors, "pins": args.pins,
        "layers": args.layers, "pattern": args.pattern, "seed": args.seed,
        "backend": "gpu" if pf.config.use_gpu else "cpu",
        "direction_mode": args.direction_mode,
        "wrong_way_cost_multiplier": config.wrong_way_cost_multiplier,
        "preferred_layer_directions": (
            config.preferred_layer_directions
        ),
        "layer_depth_bias": config.layer_depth_bias,
        "grid_pitch": config.grid_pitch,
        "hdi_stack": (
            config.hdi_stack.name if config.hdi_stack is not None else None
        ),
        "effective_max_iterations": config.max_iterations,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if memory:
        metrics["memory"] = memory
        metrics["memory"]["peak_observed_pool_total_bytes"] = max(
            sample["pool_total_bytes"] for sample in memory.values()
            if isinstance(sample, dict)
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectors", type=int, default=2)
    parser.add_argument("--pins", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--pattern", choices=["pairs", "neighbor", "bus"],
                        default="pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direction-mode",
        choices=["strict", "guided", "bidirectional"],
        default="strict",
        help="strict preferred axes, penalized wrong-way, or equal-cost H/V",
    )
    parser.add_argument(
        "--wrong-way-multiplier",
        type=float,
        default=4.0,
        help="nonpreferred-axis cost in guided mode (must be >= 1)",
    )
    parser.add_argument(
        "--layer-directions",
        help="comma-separated preferred H/V axis for every copper layer",
    )
    parser.add_argument(
        "--layer-depth-bias",
        type=float,
        default=0.0,
        help="additive cost bias per higher layer (zero disables packing)",
    )
    parser.add_argument(
        "--grid-pitch",
        type=float,
        help="routing lattice pitch in mm (default: router configuration)",
    )
    parser.add_argument(
        "--hdi-stack",
        choices=["none", "pcbway-elic"],
        default="none",
        help="explicit fabrication via topology",
    )
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--gpu", dest="use_gpu", action="store_true",
                         help="run the CUDA/CuPy path")
    backend.add_argument("--cpu", dest="use_gpu", action="store_false",
                         help="run the CPU path (default)")
    parser.set_defaults(use_gpu=False)
    parser.add_argument("-o", "--output-dir", default=str(REPO_ROOT / "benchmarks" / "results"))
    args = parser.parse_args()

    metrics = run(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / (f"{args.connectors}x{args.pins}_{args.layers}L_"
                          f"{args.pattern}_{stamp}.json")
    out_file.write_text(json.dumps(metrics, indent=2))

    c = metrics["completion"]
    copper = metrics["copper"]
    print(f"[BENCH] DONE routed={c['routed_nets']}/{c['total_nets']} "
          f"iters={metrics['convergence']['iterations']} "
          f"overuse={metrics['convergence']['overuse_total']} "
          f"layers_used={copper['layers_used']} "
          f"wirelength={copper['wirelength_mm']}mm "
          f"vias={copper['via_transitions']} "
          f"total={metrics['timings_s']['total']}s")
    print(f"[BENCH] Metrics written to {out_file}")
    return 0 if c["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
