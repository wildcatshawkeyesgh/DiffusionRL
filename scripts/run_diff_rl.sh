#!/bin/bash

# Simple background job runner
# Usage: ./run_bg.sh "your command here"

if [ $# -eq 0 ]; then
    echo "Usage: $0 'command to run'"
    echo "Example: $0 'python train.py'"
    exit 1
fi

COMMAND="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="job_${TIMESTAMP}.log"

# Run in background with nohup
nohup bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &

# Get the PID
PID=$!

echo "Job started!"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""
echo "To check status: ps -p $PID"
echo "To kill job: kill $PID"
echo "To view log: tail -f $LOG_FILE"