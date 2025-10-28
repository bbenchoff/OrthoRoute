#!/bin/bash
# Quick progress checker for routing run
# Usage: ./check_progress.sh

LOG_FILE="full_routing_run.log"

echo "=== OrthoRoute Progress Report ($(date)) ==="
echo ""

# Check if routing is still running
if ps aux | grep -v grep | grep "python main.py" > /dev/null; then
    echo "STATUS: Routing is RUNNING"
else
    echo "STATUS: Routing has COMPLETED or STOPPED"
fi
echo ""

# Get board characteristics
echo "BOARD CHARACTERISTICS:"
grep -A 10 "BOARD CHARACTERISTICS" $LOG_FILE 2>/dev/null | head -12
echo ""

# Get latest iteration summary
echo "LATEST ITERATION:"
grep "ITER [0-9]*\].*routed.*failed.*overuse" $LOG_FILE 2>/dev/null | tail -1
echo ""

# Get convergence progress
echo "CONVERGENCE PROGRESS:"
grep "ITER [0-9]*\].*routed.*failed.*overuse" $LOG_FILE 2>/dev/null | tail -10
echo ""

# Check for best result
echo "BEST RESULT:"
grep "BEST-RESULT" $LOG_FILE 2>/dev/null | tail -1
echo ""

# Timing estimates
total_nets=$(grep "nets, [0-9]* iters" $LOG_FILE 2>/dev/null | head -1 | grep -oP '\d+(?= nets)')
if [ -n "$total_nets" ]; then
    completed_iterations=$(grep "ITER [0-9]*\].*routed" $LOG_FILE 2>/dev/null | wc -l)
    echo "PROGRESS: $completed_iterations iterations completed"
    echo ""
fi

# Recent errors (excluding GPU-DEBUG)
recent_errors=$(tail -500 $LOG_FILE 2>/dev/null | grep -i "error\|exception\|failed" | grep -v "GPU-DEBUG" | grep -v "no error" | tail -5)
if [ -n "$recent_errors" ]; then
    echo "RECENT ERRORS:"
    echo "$recent_errors"
    echo ""
fi

# Show routing log file location
latest_routing_log=$(ls -t routing_log_*.txt 2>/dev/null | head -1)
if [ -n "$latest_routing_log" ]; then
    echo "DETAILED LOG: $latest_routing_log"
    echo "View with: tail -f $latest_routing_log"
fi

echo ""
echo "=== End of Progress Report ==="
