#!/bin/zsh
cd "$(dirname "$0")"
while pgrep -f run_tune.py > /dev/null; do sleep 20; done
echo "=== sweep done; evaluating best architecture in the paper's setup ==="
for P in CHB,CDX CHB,JPT; do
  echo "--- $P deeper (dilations to 512) ---"
  ./.venv/bin/python -u run_real.py --pops $P --dilations 1,2,4,8,16,32,64,128,256,512
done
echo "=== FOLLOWUP DONE ==="
