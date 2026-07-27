#!/bin/zsh
cd "$(dirname "$0")"
while pgrep -f "run_real.py|tune_followup.sh" > /dev/null; do sleep 20; done
echo "=== data-scaling curve ==="
./.venv/bin/python -u run_scaling.py --pops CHB,CDX
echo "=== SCALING DONE ==="
