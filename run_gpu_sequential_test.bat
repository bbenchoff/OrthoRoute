@echo off
REM Test GPU Sequential Mode - should achieve 82% like CPU-POC but much faster
timeout 900 python main.py --test-manhattan > test_gpu_sequential.log 2>&1
