#!/bin/bash
while true; do
    echo "=== $(date) ==="
    tail -5 test_LONG_RUN.log | grep -E "ITER|Batch"
    echo "Total lines: $(wc -l < test_LONG_RUN.log)"
    echo "Cycle errors: $(grep -c "cycle detected" test_LONG_RUN.log 2>/dev/null || echo 0)"
    echo ""
    sleep 60
done
