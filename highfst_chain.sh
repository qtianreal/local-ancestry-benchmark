#!/bin/zsh
cd "$(dirname "$0")"
while pgrep -f "run_scaling.py|scaling_chain.sh" > /dev/null; do sleep 20; done
echo "=== higher-divergence real pairs ==="
for P in FIN,TSI CHB,CEU; do
  echo "--- $P : internal methods ---"
  ./.venv/bin/python -u run_real.py --pops $P
  echo "--- $P : released tools ---"
  ./.venv/bin/python -u run_real_external.py --pops $P
done
echo "=== HIGH-FST DONE ==="
