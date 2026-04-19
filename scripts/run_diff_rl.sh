#!/bin/bash
#
# Background job runner for diffusion-RL training.
# Usage: ./run_diff_rl.sh                  # defaults to `python -u train.py`
#        ./run_diff_rl.sh "python -u foo.py --flag"
#
# Why `-u` / PYTHONUNBUFFERED: when stdout is redirected to a file (not a
# TTY), Python block-buffers output, so `tail -f log` shows nothing until
# the buffer fills (~4KB) or the process exits. `-u` forces line-buffered
# stdout/stderr so log lines appear as soon as they're printed.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    COMMAND="python -u $SCRIPT_DIR/train.py"
else
    COMMAND="$1"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SCRIPT_DIR/job_${TIMESTAMP}.log"

# PYTHONUNBUFFERED=1 forces Python to line-buffer stdout/stderr when the
# process isn't attached to a TTY. `stdbuf` does the same for non-Python
# children (C/shell tools), but it's optional — not installed by default on
# macOS.
if command -v stdbuf >/dev/null 2>&1; then
    PYTHONUNBUFFERED=1 nohup stdbuf -oL -eL bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &
else
    PYTHONUNBUFFERED=1 nohup bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &
fi
PID=$!

echo "Job started!"
echo "  PID: $PID"
echo "  Log: $LOG_FILE"
echo "  Cmd: $COMMAND"
echo ""
echo "To check status: ps -p $PID"
echo "To kill job:     kill $PID"
echo "To view log:     tail -f \"$LOG_FILE\""
