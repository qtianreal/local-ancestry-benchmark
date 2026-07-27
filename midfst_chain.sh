#!/bin/zsh
cd "$(dirname "$0")"
echo "=== filling the 0.012-0.109 gap ==="
for P in TSI,PJL CEU,GIH CHB,BEB CHB,GIH; do
  echo "--- $P ---"
  ./.venv/bin/python -u run_real.py --pops $P
  ./.venv/bin/python -u run_real_external.py --pops $P
done
echo "=== MID-FST DONE ==="
