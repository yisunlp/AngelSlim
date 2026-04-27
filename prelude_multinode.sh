#!/usr/bin/env bash
# Per-node prelude: clean up stale workers from previous runs.
# Run on every node BEFORE the DeepSpeed launcher kicks off.
pkill -f "python3 -" 2>/dev/null || true
pkill -f "tools/run.py" 2>/dev/null || true
echo "[prelude] $(hostname -s) cleaned stale processes"
