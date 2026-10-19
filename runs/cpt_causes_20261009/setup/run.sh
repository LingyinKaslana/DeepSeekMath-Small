#!/usr/bin/env bash
set -euo pipefail
CPT_PROJECT=.
CPT_SETUP="$CPT_PROJECT/runs/cpt_causes_20261009/setup"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$CPT_SETUP/suite_pid"
exec python "$CPT_SETUP/run_hyperparameters.py" > "$CPT_SETUP/hyperparameter_suite.log" 2>&1
