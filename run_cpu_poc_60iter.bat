@echo off
set ORTHO_CPU_ONLY=1
timeout 3600 python main.py --test-manhattan > test_cpu_poc_60iter.log 2>&1
