#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=.
LAUNCH_ROOT="$PROJECT_ROOT/runs/cpt_native_long_launch_20261009"
source "$PROJECT_ROOT/activate.sh"
exec python "$LAUNCH_ROOT/source/train_cpt.py" \
  --backend-config "$LAUNCH_ROOT/backend_config.json" \
  --data "$PROJECT_ROOT/data/processed/openwebmath_5b_s42" \
  --output "$PROJECT_ROOT/runs/cpt_native_long_20261009/migration_smoke" \
  --resume "$PROJECT_ROOT/runs/cpt_10b_20261009_after_native_compare/stopped_checkpoint" \
  --allow-native-backend-migration --stop-after 181654 \
  --epochs 2 --lr 1e-5 --seed 42 --log-every 1 \
  --eval-every 1000 --checkpoint-every 5000 --keep-checkpoints 3 \
  --train-probe-blocks 256 --train-probe-seed 20261009
