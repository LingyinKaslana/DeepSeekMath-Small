#!/usr/bin/env bash
set -euo pipefail
CPT_PROJECT=.
CPT_DIAG="$CPT_PROJECT/runs/cpt_seen_unseen_lr3e6_20261010"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$CPT_DIAG/coordinator_pid"
exec > "$CPT_DIAG/evaluations.log" 2>&1
exec python "$CPT_DIAG/tools/run_evaluations.py"
