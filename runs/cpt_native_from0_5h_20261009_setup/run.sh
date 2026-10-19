#!/usr/bin/env bash
set -euo pipefail

CPT_PROJECT=.
CPT_SETUP="$CPT_PROJECT/runs/cpt_native_from0_5h_20261009_setup"
CPT_RUN="$CPT_PROJECT/runs/cpt_native_from0_5h_20261009"
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
    python "$CPT_SETUP/analysis_tools/observation_status.py" \
        --setup "$CPT_SETUP" --run "$CPT_RUN" \
        --stage "$CPT_STAGE" --exit-code "$CPT_EXIT_CODE"
    exit "$CPT_EXIT_CODE"
}
trap on_signal TERM INT
trap on_exit EXIT

# Fresh base and fresh Adam state. Wall-time bounds execution only; the original
# two-epoch, 10B-token learning-rate schedule is intentionally unchanged.
python "$CPT_SETUP/source/train_cpt.py" \
    --backend-config "$CPT_SETUP/backend_config.json" \
    --data "$CPT_PROJECT/data/processed/openwebmath_5b_s42" \
    --output "$CPT_RUN" \
    --epochs 2 --lr 1e-5 --seed 42 \
    --log-every 100 --eval-every 1000 \
    --checkpoint-every 5000 --keep-checkpoints 3 \
    --train-probe-blocks 256 --train-probe-seed 20261009 \
    --max-seconds 18000 &
CPT_TRAIN_PID=$!
printf '%s\n' "$CPT_TRAIN_PID" > "$CPT_SETUP/trainer_pid"
set +e
wait "$CPT_TRAIN_PID"
CPT_TRAIN_EXIT=$?
# A shell signal can interrupt wait while the trainer is saving its checkpoint.
if kill -0 "$CPT_TRAIN_PID" 2>/dev/null; then
    wait "$CPT_TRAIN_PID"
    CPT_TRAIN_EXIT=$?
fi
set -e
if [[ "$CPT_TRAIN_EXIT" -ne 0 ]]; then
    exit "$CPT_TRAIN_EXIT"
fi

CPT_STAGE=checkpoint_verification
python "$CPT_SETUP/source/verify_cpt_checkpoint.py" "$CPT_RUN/stopped_checkpoint" \
    --output "$CPT_SETUP/final_checkpoint_verification.json" \
    > "$CPT_SETUP/final_checkpoint_verification.log" 2>&1

CPT_STAGE=curve_report
"$CPT_PROJECT/.venv-plot/bin/python" "$CPT_SETUP/analysis_tools/plot_cpt_native_from0.py" \
    --native-run "$CPT_RUN" --fast-run "$CPT_PROJECT/runs/cpt_10b_20261008" \
    --output "$CPT_SETUP/report" > "$CPT_SETUP/final_report.log" 2>&1

CPT_STAGE=finished
# The old 6.105B continuation remains paused. This observation is separate.
