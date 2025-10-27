#!/bin/bash
# PathFinder Parameter Sweep - Quick Start
# ==========================================
# This script runs the overnight parameter sweep for PathFinder convergence tuning.
#
# What it does:
# - Tests 12 parameter configurations (A1-A6 Greedy, B1-B6 Penalized)
# - Runs in priority order: B4, B1, A4, A1, then others
# - Applies early stopping to save time
# - Tracks all metrics (overuse, layer distribution, timing)
# - Saves results to parameter_sweep_results/
# - Generates summary report
#
# Expected runtime: ~2 hours (varies with early stopping)
#
# Resume capability: If interrupted, run: python parameter_sweep.py --resume

echo "================================================================================"
echo "PathFinder Parameter Sweep - Overnight Run"
echo "================================================================================"
echo ""
echo "This will run 12 parameter configurations to find optimal convergence settings."
echo "Expected runtime: ~2 hours"
echo ""
echo "Results will be saved to: parameter_sweep_results/"
echo "  - sweep_results.json (complete data)"
echo "  - sweep_summary.txt (human-readable report)"
echo "  - run_*.log (individual run logs)"
echo ""
echo "You can interrupt with Ctrl+C and resume later with:"
echo "  python parameter_sweep.py --resume"
echo ""
echo "================================================================================"
echo ""
read -p "Press Enter to start sweep..."

echo ""
echo "Starting parameter sweep..."
echo ""

python parameter_sweep.py

echo ""
echo "================================================================================"
echo "Sweep Complete!"
echo "================================================================================"
echo ""
echo "View results:"
echo "  cat parameter_sweep_results/sweep_summary.txt"
echo ""
echo "Generate plots:"
echo "  python plot_sweep_results.py"
echo ""
