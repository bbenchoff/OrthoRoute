@echo off
REM Run GPU Sequential for ALL 60 iterations (True PathFinder)
set SEQUENTIAL_ALL=1
set MAX_ITERATIONS=60
timeout 7200 python main.py --test-manhattan > test_sequential_ALL_60iter.log 2>&1
