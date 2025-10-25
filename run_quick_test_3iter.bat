@echo off
REM Quick test with 3 iterations to verify optimizations
REM Expected: Similar success rate (88-92%), improved speed
set PATHFINDER_ITERATIONS=3
timeout 300 python main.py --test-manhattan > test_quick_3iter.log 2>&1
