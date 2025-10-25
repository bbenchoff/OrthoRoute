@echo off
REM Test GPU Sequential Mode with pool reset fix
set GPU_SEQUENTIAL=1
timeout 900 python main.py --test-manhattan > test_gpu_seq_FIXED.log 2>&1
