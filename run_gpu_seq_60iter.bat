@echo off
REM Run GPU Sequential Mode with extended iterations for convergence testing
set GPU_SEQUENTIAL=1
set MAX_ITERATIONS=60
timeout 7200 python main.py --test-manhattan > test_gpu_seq_60iter.log 2>&1
