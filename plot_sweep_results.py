#!/usr/bin/env python3
"""
Plot Parameter Sweep Results
=============================

Generates visualizations from parameter sweep results for analysis.

Usage:
    python plot_sweep_results.py [--output-dir DIR]

Generates:
- Convergence comparison plot (all runs)
- Best run detailed plot
- Parameter impact bar charts
- Layer distribution heatmap
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not installed. Install with: pip install matplotlib")


BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "parameter_sweep_results" / "sweep_results.json"


def load_results() -> Dict[str, Any]:
    """Load sweep results from JSON file."""
    if not RESULTS_FILE.exists():
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        print("Run parameter_sweep.py first to generate results.")
        return {}

    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def plot_convergence_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot convergence curves for all successful runs."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    successful = {k: v for k, v in results.items() if v.get('success', False)}

    for run_id, result in successful.items():
        iterations = result.get('iterations', [])
        if not iterations:
            continue

        iter_nums = [it['iteration'] for it in iterations]
        overuses = [it['overuse'] for it in iterations]

        # Color by start mode
        color = 'blue' if result['config']['start_mode'] == 'greedy' else 'red'
        alpha = 0.7 if run_id in ['B4', 'B1', 'A4', 'A1'] else 0.3  # Highlight priority runs

        ax.plot(iter_nums, overuses, label=run_id, color=color, alpha=alpha, linewidth=2 if alpha > 0.5 else 1)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Overuse', fontsize=12)
    ax.set_title('PathFinder Convergence Comparison - All Runs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_ylim(bottom=0)

    # Add convergence target line
    ax.axhline(y=1000, color='green', linestyle='--', linewidth=2, label='Target (1000)', alpha=0.7)

    plt.tight_layout()
    output_file = output_dir / "convergence_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved: {output_file}")


def plot_best_run_detailed(results: Dict[str, Any], output_dir: Path):
    """Plot detailed analysis of best run."""
    if not HAS_MATPLOTLIB:
        return

    successful = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful:
        return

    # Find best run
    best_run_id = min(successful.keys(), key=lambda k: successful[k]['best_overuse'])
    best_result = successful[best_run_id]
    iterations = best_result.get('iterations', [])

    if not iterations:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Best Run Detailed Analysis: {best_run_id}', fontsize=16, fontweight='bold')

    # Plot 1: Overuse progression
    ax = axes[0, 0]
    iter_nums = [it['iteration'] for it in iterations]
    overuses = [it['overuse'] for it in iterations]
    ax.plot(iter_nums, overuses, 'b-', linewidth=2, label='Total Overuse')
    ax.axhline(y=best_result['best_overuse'], color='green', linestyle='--', label=f"Best: {best_result['best_overuse']}")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Overuse')
    ax.set_title('Convergence Progression')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Via overuse percentage
    ax = axes[0, 1]
    via_overuse = [it['via_overuse_pct'] for it in iterations]
    ax.plot(iter_nums, via_overuse, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Via Overuse (%)')
    ax.set_title('Via Overuse Percentage')
    ax.grid(True, alpha=0.3)

    # Plot 3: Layer distribution (top 3 layers over time)
    ax = axes[1, 0]
    if iterations[0].get('top3_layers'):
        layer_data = {}
        for it in iterations:
            for layer_num, _, pct in it.get('top3_layers', []):
                if layer_num not in layer_data:
                    layer_data[layer_num] = []
                layer_data[layer_num].append((it['iteration'], pct))

        for layer_num, data in sorted(layer_data.items())[:5]:  # Top 5 layers
            iters, pcts = zip(*data)
            ax.plot(iters, pcts, marker='o', label=f'Layer {layer_num}', linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overuse Percentage (%)')
        ax.set_title('Top Layer Distribution Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 4: Iteration timing
    ax = axes[1, 1]
    times = [it['time_sec'] for it in iterations if it.get('time_sec', 0) > 0]
    if times:
        ax.plot(range(len(times)), times, 'g-', linewidth=2)
        ax.axhline(y=sum(times)/len(times), color='orange', linestyle='--', label=f"Avg: {sum(times)/len(times):.1f}s")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Iteration Timing')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_file = output_dir / f"best_run_{best_run_id}_detailed.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved: {output_file}")


def plot_parameter_impact(results: Dict[str, Any], output_dir: Path):
    """Plot parameter impact analysis."""
    if not HAS_MATPLOTLIB:
        return

    successful = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful:
        return

    # Analyze parameter impact
    param_impact = {}
    for param_name in ['initial_pres', 'layer_bias_max', 'via_depart_max', 'start_mode']:
        param_impact[param_name] = {}
        for run_id, result in successful.items():
            value = result['config'][param_name]
            if value not in param_impact[param_name]:
                param_impact[param_name][value] = []
            param_impact[param_name][value].append(result['best_overuse'])

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parameter Impact on Best Overuse', fontsize=16, fontweight='bold')

    param_names = ['initial_pres', 'layer_bias_max', 'via_depart_max', 'start_mode']
    param_labels = ['Initial Pressure Factor', 'Max Layer Bias', 'Max Via Departure', 'Start Mode']

    for idx, (param_name, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx // 2, idx % 2]

        values = sorted(param_impact[param_name].keys(), key=lambda x: (isinstance(x, str), x))
        avgs = [sum(param_impact[param_name][v]) / len(param_impact[param_name][v]) for v in values]
        counts = [len(param_impact[param_name][v]) for v in values]

        x_labels = [f"{v}\n(n={c})" for v, c in zip(values, counts)]

        bars = ax.bar(range(len(values)), avgs, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Average Best Overuse')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, avg in zip(bars, avgs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(avg)}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / "parameter_impact.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved: {output_file}")


def plot_ranking_summary(results: Dict[str, Any], output_dir: Path):
    """Plot ranking summary bar chart."""
    if not HAS_MATPLOTLIB:
        return

    successful = {k: v for k, v in results.items() if v.get('success', False)}
    if not successful:
        return

    # Sort by best overuse
    ranked = sorted(successful.items(), key=lambda x: x[1]['best_overuse'])

    fig, ax = plt.subplots(figsize=(10, 6))

    run_ids = [run_id for run_id, _ in ranked]
    best_overuses = [result['best_overuse'] for _, result in ranked]
    final_overuses = [result['final_overuse'] for _, result in ranked]

    x = range(len(run_ids))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], best_overuses, width, label='Best Overuse', color='green', alpha=0.7)
    bars2 = ax.bar([i + width/2 for i in x], final_overuses, width, label='Final Overuse', color='orange', alpha=0.7)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Overuse', fontsize=12)
    ax.set_title('All Configurations Ranked by Best Overuse', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add convergence target line
    ax.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Target (1000)', alpha=0.7)

    plt.tight_layout()
    output_file = output_dir / "ranking_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved: {output_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Plot parameter sweep results")
    parser.add_argument('--output-dir', default='parameter_sweep_results',
                       help='Output directory for plots (default: parameter_sweep_results)')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib is required for plotting")
        print("Install with: pip install matplotlib")
        return 1

    print("Loading results...")
    results = load_results()

    if not results:
        print("[ERROR] No results to plot")
        return 1

    successful = {k: v for k, v in results.items() if v.get('success', False)}
    print(f"Found {len(successful)} successful runs out of {len(results)} total")

    if not successful:
        print("[ERROR] No successful runs to plot")
        return 1

    print("\nGenerating plots...")
    plot_convergence_comparison(results, output_dir)
    plot_best_run_detailed(results, output_dir)
    plot_parameter_impact(results, output_dir)
    plot_ranking_summary(results, output_dir)

    print(f"\n[SUCCESS] All plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - convergence_comparison.png   : All runs overlaid")
    print("  - best_run_*_detailed.png      : Best run deep dive")
    print("  - parameter_impact.png         : Parameter sensitivity analysis")
    print("  - ranking_summary.png          : Final ranking bar chart")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
