@echo off
REM PathFinder Parameter Sweep - Quick Start
REM ==========================================
REM This batch file runs the overnight parameter sweep for PathFinder convergence tuning.
REM
REM What it does:
REM - Tests 12 parameter configurations (A1-A6 Greedy, B1-B6 Penalized)
REM - Runs in priority order: B4, B1, A4, A1, then others
REM - Applies early stopping to save time
REM - Tracks all metrics (overuse, layer distribution, timing)
REM - Saves results to parameter_sweep_results/
REM - Generates summary report
REM
REM Expected runtime: ~2 hours (varies with early stopping)
REM
REM Resume capability: If interrupted, run: python parameter_sweep.py --resume

echo ================================================================================
echo PathFinder Parameter Sweep - Overnight Run
echo ================================================================================
echo.
echo This will run 12 parameter configurations to find optimal convergence settings.
echo Expected runtime: ~2 hours
echo.
echo Results will be saved to: parameter_sweep_results/
echo   - sweep_results.json (complete data)
echo   - sweep_summary.txt (human-readable report)
echo   - run_*.log (individual run logs)
echo.
echo You can interrupt with Ctrl+C and resume later with:
echo   python parameter_sweep.py --resume
echo.
echo ================================================================================
echo.
pause

echo.
echo Starting parameter sweep...
echo.

python parameter_sweep.py

echo.
echo ================================================================================
echo Sweep Complete!
echo ================================================================================
echo.
echo View results:
echo   type parameter_sweep_results\sweep_summary.txt
echo.
echo Generate plots:
echo   python plot_sweep_results.py
echo.
pause
