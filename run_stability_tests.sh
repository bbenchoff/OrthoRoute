#!/bin/bash

echo "Starting stability test run at $(date)"
echo "Target: 50 successful GPU routing tests"
echo "============================================"

PASS=0
FAIL=0
MAX_RUNS=50

for i in $(seq 1 $MAX_RUNS); do
    echo ""
    echo "----------------------------------------"
    echo "Run #$i of $MAX_RUNS - $(date)"
    echo "----------------------------------------"

    timeout 300 python test_gpu_fast.py
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        PASS=$((PASS+1))
        echo "✓ Run #$i: PASS (Total: $PASS pass, $FAIL fail)"
    else
        FAIL=$((FAIL+1))
        echo "✗ Run #$i: FAIL (Total: $PASS pass, $FAIL fail)"
    fi

    # Quick stats
    SUCCESS_RATE=$(echo "scale=1; 100 * $PASS / ($PASS + $FAIL)" | bc)
    echo "Current success rate: $SUCCESS_RATE%"

    # Stop early if we hit critical failures
    if [ $FAIL -gt 5 ]; then
        echo "ERROR: Too many failures ($FAIL), stopping test"
        break
    fi
done

echo ""
echo "============================================"
echo "STABILITY TEST COMPLETE"
echo "============================================"
echo "Total runs: $((PASS+FAIL))"
echo "Passed: $PASS"
echo "Failed: $FAIL"
SUCCESS_RATE=$(echo "scale=1; 100 * $PASS / ($PASS + $FAIL)" | bc)
echo "Success rate: $SUCCESS_RATE%"
echo "Completed at: $(date)"
