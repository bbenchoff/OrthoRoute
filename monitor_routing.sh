#!/bin/bash
# Monitoring script for OrthoRoute routing progress
# Checks progress every 5 minutes and reports status

LOG_FILE="cpu_test_fixed.txt"
MONITOR_LOG="routing_monitor.log"

echo "=== OrthoRoute Routing Monitor Started at $(date) ===" | tee -a $MONITOR_LOG

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if routing is still running
    if ! ps aux | grep -v grep | grep "python main.py" > /dev/null; then
        echo "[$timestamp] Routing process completed or crashed" | tee -a $MONITOR_LOG
        break
    fi

    # Get latest routing progress
    latest_net=$(tail -100 $LOG_FILE 2>/dev/null | grep "Net [0-9]*/512" | tail -1)
    latest_iter=$(tail -100 $LOG_FILE 2>/dev/null | grep "ITER [0-9]*\].*routed" | tail -1)

    # Report progress
    if [ -n "$latest_net" ]; then
        echo "[$timestamp] $latest_net" | tee -a $MONITOR_LOG
    fi

    if [ -n "$latest_iter" ]; then
        echo "[$timestamp] $latest_iter" | tee -a $MONITOR_LOG
    fi

    # Check for errors
    errors=$(tail -100 $LOG_FILE 2>/dev/null | grep -i "error\|exception\|failed" | grep -v "GPU-DEBUG" | tail -3)
    if [ -n "$errors" ]; then
        echo "[$timestamp] Recent errors detected:" | tee -a $MONITOR_LOG
        echo "$errors" | tee -a $MONITOR_LOG
    fi

    # Sleep 5 minutes
    sleep 300
done

echo "=== OrthoRoute Routing Monitor Ended at $(date) ===" | tee -a $MONITOR_LOG
