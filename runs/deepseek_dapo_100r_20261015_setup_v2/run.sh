#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
export PYTHONUNBUFFERED=1
DAPO_SETUP=runs/deepseek_dapo_100r_20261015_setup_v2
DAPO_RUN=runs/deepseek_dapo_100r_20261015
DAPO_CHECKPOINT=runs/deepseek_sft_gsm_math_20261015/best_checkpoint
DAPO_DATA=data/processed/sft_gsm_math_4096_s42_20261015
DAPO_VENDOR=runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math
DAPO_PHASE=cpu_checks
on_exit() {
    DAPO_EXIT=$?
    trap - EXIT
    if [ "$DAPO_EXIT" -ne 0 ]; then
        python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status failed --exit-code "$DAPO_EXIT"
    fi
    exit "$DAPO_EXIT"
}
trap on_exit EXIT
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/test_dapo_math.py" > "$DAPO_SETUP/cpu_checks.log" 2>&1
DAPO_PHASE=gpu_smoke
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/dapo_math.py" --checkpoint "$DAPO_CHECKPOINT" --data "$DAPO_DATA" \
    --output "$DAPO_SETUP/smoke" --rounds 2 --groups 4 --samples 8 --rollout-prompts 4 \
    --update-batch 8 --validation-size 4 --diagnostic-prompts 8 --smoke \
    > "$DAPO_SETUP/smoke.log" 2>&1
python "$DAPO_SETUP/smoke_gate.py"
DAPO_PHASE=formal_diagnostic_and_training
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/dapo_math.py" --checkpoint "$DAPO_CHECKPOINT" --data "$DAPO_DATA" \
    --output "$DAPO_RUN" --rounds 100 --groups 32 --samples 8 --rollout-prompts 16 \
    --update-batch 64 --validation-size 256 --diagnostic-prompts 256 \
    > "$DAPO_SETUP/training.log" 2>&1
DAPO_PHASE=full_benchmark_generation
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/run_math_inference.py" --checkpoint "$DAPO_RUN/best_checkpoint" \
    --model-kind rl --reference-run runs/deepseek_coder_10b_inference_20261015 \
    --vendor "$DAPO_VENDOR" --output "$DAPO_SETUP/evaluation" --batch-size 256 --max-new-tokens 1024 \
    > "$DAPO_SETUP/evaluation_generation.log" 2>&1
DAPO_PHASE=full_benchmark_scoring
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/source/score_math_benchmarks.py" --run "$DAPO_SETUP/evaluation" --vendor "$DAPO_VENDOR" \
    > "$DAPO_SETUP/evaluation_scoring.log" 2>&1
DAPO_PHASE=final_verification
python "$DAPO_SETUP/control.py" --phase "$DAPO_PHASE" --status running
python "$DAPO_SETUP/report.py" > "$DAPO_SETUP/report.log" 2>&1
python "$DAPO_SETUP/control.py" --phase finished --status complete_verified --exit-code 0
