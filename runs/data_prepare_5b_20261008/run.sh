#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=.
LAUNCH_ROOT="$PROJECT_ROOT/runs/data_prepare_5b_20261008"
source "$PROJECT_ROOT/activate.sh"
printf '%s\n' "$$" > "$LAUNCH_ROOT/pid"
exec > "$LAUNCH_ROOT/prepare.log" 2>&1
exec python "$LAUNCH_ROOT/source/prepare_openwebmath_large.py" \
 --source-manifest "$PROJECT_ROOT/data/raw/openwebmath_20261008/manifest.json" \
 --pilot-data "$PROJECT_ROOT/data/processed/openwebmath_10m_s42" \
 --output "$PROJECT_ROOT/data/processed/openwebmath_5b_s42" \
 --train-tokens 5000000000 --validation-tokens 2097152 \
 --workers 12 --tokenizer-threads 4 "$@"
