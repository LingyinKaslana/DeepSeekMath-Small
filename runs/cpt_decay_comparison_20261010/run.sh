#!/usr/bin/env bash
set -euo pipefail
CPT_PROJECT=.
CPT_EXPERIMENT="$CPT_PROJECT/runs/cpt_decay_comparison_20261010"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$CPT_EXPERIMENT/coordinator_pid"
exec python "$CPT_EXPERIMENT/tools/run_comparison.py" > "$CPT_EXPERIMENT/coordinator.log" 2>&1
