#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
export PYTHONUNBUFFERED=1
DAPO_SETUP=runs/deepseek_dapo_10k_20261018_recovery_setup
DAPO_RUN=runs/deepseek_dapo_10k_20261016
DAPO_VENDOR=runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math
on_exit() {
    DAPO_EXIT=$?
    trap - EXIT
    if [ "$DAPO_EXIT" -ne 0 ]; then
        python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status failed --exit-code "$DAPO_EXIT"
    fi
    exit "$DAPO_EXIT"
}
trap on_exit EXIT
DAPO_POLICY=history_rank
DAPO_PHASE=resumed_optimized_training
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/dapo_math.py" --checkpoint runs/deepseek_sft_gsm_math_20261015/best_checkpoint \
    --data data/processed/sft_gsm_math_4096_s42_20261015 --output "$DAPO_RUN" \
    --rounds 10000 --groups 32 --samples 8 --rollout-prompts 32 --micro-batch 16 \
    --update-batch 64 --validation-size 256 --diagnostic-prompts 0 --compact-decode \
    --resume "$DAPO_SETUP/resume_checkpoint" --allow-sampling-change --candidate-group-budget 2048 --recovery-every 10 --sampling-policy "$DAPO_POLICY" \
    --sampling-history "$DAPO_SETUP/bootstrap/history_audit.json" --stop-file "$DAPO_SETUP/save_and_stop" \
    --save-every 100 --eval-every 100 --retain-every-checkpoint --verify-resume > "$DAPO_SETUP/training.log" 2>&1
if [ -f "$DAPO_RUN/cooperative_stop.json" ]; then
    python "$DAPO_SETUP/control.py" --phase checkpointed_stop --status stopped --exit-code 0
    exit 0
fi
DAPO_PHASE=full_benchmark_generation
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/run_math_inference.py" --checkpoint "$DAPO_RUN/best_checkpoint" \
    --model-kind rl --reference-run runs/deepseek_coder_10b_inference_20261015 \
    --vendor "$DAPO_VENDOR" --output "$DAPO_SETUP/evaluation" --batch-size 256 --max-new-tokens 1024 \
    > "$DAPO_SETUP/evaluation_generation.log" 2>&1
DAPO_PHASE=full_benchmark_scoring
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/score_math_benchmarks.py" --run "$DAPO_SETUP/evaluation" --vendor "$DAPO_VENDOR" > "$DAPO_SETUP/evaluation_scoring.log" 2>&1
DAPO_PHASE=final_verification
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/report.py" > "$DAPO_SETUP/report.log" 2>&1
python "$DAPO_SETUP/control.py" --phase finished --status complete_verified --exit-code 0
