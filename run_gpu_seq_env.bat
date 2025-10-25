@echo off
REM Force GPU Sequential Mode via environment variable
set GPU_SEQUENTIAL=1
timeout 900 python main.py --test-manhattan > test_gpu_seq_final.log 2>&1
