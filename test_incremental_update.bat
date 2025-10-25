@echo off
REM Test incremental cost update feature with 3 iterations
set PATHFINDER_ITERATIONS=3
set INCREMENTAL_COST_UPDATE=1
timeout 300 python main.py --test-manhattan > test_incremental.log 2>&1
