#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=.
LAUNCH_ROOT="$PROJECT_ROOT/runs/cpt_10b_20261009_after_compare_launch"
source "$PROJECT_ROOT/activate.sh"
printf '%s\n' "$$" > "$LAUNCH_ROOT/pid"
exec > "$LAUNCH_ROOT/training.log" 2>&1
exec python "$LAUNCH_ROOT/source/train_cpt.py" \
 --backend-config "$LAUNCH_ROOT/backend_config.json" \
 --data "$PROJECT_ROOT/data/processed/openwebmath_5b_s42" \
 --output "$PROJECT_ROOT/runs/cpt_10b_20261009_after_native_compare" \
 --resume "$PROJECT_ROOT/runs/cpt_10b_20261009_resumed/stopped_checkpoint" \
 --retry-native-on-nonfinite \
 --epochs 2 --lr 1e-5 --seed 42 \
 --log-every 100 --eval-every 5000 --checkpoint-every 5000 --keep-checkpoints 3 "$@"
