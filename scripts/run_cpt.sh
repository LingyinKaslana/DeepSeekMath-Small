#!/usr/bin/env bash
# Example: bash scripts/run_cpt.sh --output runs/my_cpt
# --stop-after preserves the schedule; --resume accepts a stopped checkpoint.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$PROJECT_ROOT/activate.sh"
exec python "$PROJECT_ROOT/scripts/train_cpt.py" \
  --backend-config "$PROJECT_ROOT/configs/cpt_backend_native.json" \
  --data "$PROJECT_ROOT/data/processed/openwebmath_10m_s42" \
  "$@"
