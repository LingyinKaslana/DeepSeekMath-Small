#!/usr/bin/env bash
# Train two complete epochs over the prepared 5B-token OpenWebMath sample.
# Output must be new; --resume restores the original schedule in a new output.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT_ROOT/activate.sh"
exec python "$PROJECT_ROOT/scripts/train_cpt.py" \
  --backend-config "$PROJECT_ROOT/configs/cpt_backend_native.json" \
  --data "$PROJECT_ROOT/data/processed/openwebmath_5b_s42" \
  --epochs 2 \
  --lr 1e-5 \
  --seed 42 \
  --log-every 100 \
  --eval-every 1000 \
  --train-probe-blocks 256 \
  --train-probe-seed 20261009 \
  --checkpoint-every 5000 \
  --keep-checkpoints 3 \
  "$@"
