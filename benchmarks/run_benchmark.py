"""CPU-only routing benchmark on a synthetic backplane.

Usage:
    python benchmarks/run_benchmark.py --connectors 2 --pins 16 --layers 4
    python benchmarks/run_benchmark.py --connectors 4 --pins 40 --layers 8 --pattern pairs

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
    config.portal_x_snap_max = 0.75  # pads sit half a pitch off-grid (see conftest)
    pf = UnifiedPathFinder(config=config, use_gpu=False)

    timings = {}

    def phase(name, fn):
        t0 = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - t0
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
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectors", type=int, default=2)
    parser.add_argument("--pins", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--pattern", choices=["pairs", "neighbor", "bus"],
                        default="pairs")
    parser.add_argument("--seed", type=int, default=42)
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
