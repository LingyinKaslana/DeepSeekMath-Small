#!/usr/bin/env bash
set -euo pipefail

CPT_PROJECT=.
CPT_SETUP="$CPT_PROJECT/runs/cpt_10b_lr3e6_20261009_launch"
CPT_RUN="$CPT_PROJECT/runs/cpt_10b_lr3e6_20261009"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$CPT_SETUP/wrapper_pid"
exec > "$CPT_SETUP/training.log" 2>&1

CPT_STAGE=training
CPT_TRAIN_PID=""
on_signal() {
    if [[ -n "$CPT_TRAIN_PID" ]] && kill -0 "$CPT_TRAIN_PID" 2>/dev/null; then
        kill -TERM "$CPT_TRAIN_PID"
    fi
}
on_exit() {
    CPT_EXIT_CODE=$?
    trap - EXIT
    python "$CPT_SETUP/tools/record_run_status.py" \
        --setup "$CPT_SETUP" --run "$CPT_RUN" \
        --stage "$CPT_STAGE" --exit-code "$CPT_EXIT_CODE"
    exit "$CPT_EXIT_CODE"
}
trap on_signal TERM INT
trap on_exit EXIT

# A fresh base and fresh Adam state; complete the fixed two-epoch 10B schedule.
python "$CPT_SETUP/source/train_cpt.py" \
    --backend-config "$CPT_SETUP/backend_config.json" \
    --data "$CPT_PROJECT/data/processed/openwebmath_5b_s42" \
    --output "$CPT_RUN" \
    --epochs 2 --lr 3e-6 --seed 42 \
    --log-every 100 --eval-every 1000 \
    --checkpoint-every 5000 --keep-checkpoints 64 \
    --train-probe-blocks 256 --train-probe-seed 20261009 \
    --retry-native-on-nonfinite &
CPT_TRAIN_PID=$!
printf '%s\n' "$CPT_TRAIN_PID" > "$CPT_SETUP/trainer_pid"
set +e
wait "$CPT_TRAIN_PID"
CPT_TRAIN_EXIT=$?
# A shell signal can interrupt wait while the trainer is saving its checkpoint.
while kill -0 "$CPT_TRAIN_PID" 2>/dev/null; do
    wait "$CPT_TRAIN_PID"
    CPT_TRAIN_EXIT=$?
done
set -e
if [[ "$CPT_TRAIN_EXIT" -ne 0 ]]; then
    exit "$CPT_TRAIN_EXIT"
fi

CPT_STAGE=checkpoint_verification
CPT_CHECKPOINT=$(python - "$CPT_RUN/result.json" <<'PY'
import json
import sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
if result.get("status") not in {"complete", "stopped"}:
    raise RuntimeError("Training did not publish a successful final/stopped checkpoint")
print(result["checkpoint"])
PY
)
python "$CPT_SETUP/source/verify_cpt_checkpoint.py" "$CPT_CHECKPOINT" \
    --output "$CPT_SETUP/final_checkpoint_verification.json" \
    > "$CPT_SETUP/final_checkpoint_verification.log" 2>&1

CPT_STAGE=curve_report
"$CPT_PROJECT/.venv-plot/bin/python" "$CPT_SETUP/tools/plot_cpt_validation.py" \
    --run "$CPT_RUN" --output "$CPT_SETUP/report" \
    > "$CPT_SETUP/final_report.log" 2>&1
CPT_STAGE=finished
