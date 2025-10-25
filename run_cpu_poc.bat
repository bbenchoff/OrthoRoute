@echo off
REM Phase 1 CPU Proof-of-Concept Test
REM Force CPU-only sequential routing with per-net cost updates

echo ================================================================================
echo PHASE 1: CPU PROOF-OF-CONCEPT TEST
echo ================================================================================
echo This will run CPU-only sequential routing with cost updates after EACH net
echo Expected time: 10-30 minutes (slower than GPU but proves the hypothesis)
echo ================================================================================

set ORTHO_CPU_ONLY=1
python main.py --test-manhattan > cpu_poc.log 2>&1

echo.
echo ================================================================================
echo Test completed! Results saved to cpu_poc.log
echo ================================================================================
