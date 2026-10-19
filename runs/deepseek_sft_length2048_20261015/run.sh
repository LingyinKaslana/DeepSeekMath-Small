#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
export PYTHONUNBUFFERED=1
EXTENSION_SETUP=runs/deepseek_sft_length2048_20261015
EXTENSION_PHASE=startup
on_exit() {
    EXTENSION_EXIT=$?
    trap - EXIT
    if [ "$EXTENSION_EXIT" -ne 0 ];then
        python "$EXTENSION_SETUP/control.py" --phase "$EXTENSION_PHASE" --status failed --exit-code "$EXTENSION_EXIT"
    fi
    exit "$EXTENSION_EXIT"
}
trap on_exit EXIT
for EXTENSION_MODEL in sft;do
    EXTENSION_SOURCE=runs/deepseek_sft_gsm_math_20261015_setup/evaluation
    EXTENSION_PHASE="${EXTENSION_MODEL}_generation"
    python "$EXTENSION_SETUP/control.py" --phase "$EXTENSION_PHASE" --status running
    python "$EXTENSION_SETUP/source/extend_math_generations.py" \
      --source-run "$EXTENSION_SOURCE" --vendor runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math \
      --output "$EXTENSION_SETUP/$EXTENSION_MODEL" --batch-size 128 --resume \
      > "$EXTENSION_SETUP/${EXTENSION_MODEL}_generation.log" 2>&1
    EXTENSION_PHASE="${EXTENSION_MODEL}_scoring"
    python "$EXTENSION_SETUP/control.py" --phase "$EXTENSION_PHASE" --status running
    python "$EXTENSION_SETUP/source/score_math_benchmarks.py" \
      --run "$EXTENSION_SETUP/$EXTENSION_MODEL" --vendor runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math \
      > "$EXTENSION_SETUP/${EXTENSION_MODEL}_scoring.log" 2>&1
done
EXTENSION_PHASE=verification_and_comparison
python "$EXTENSION_SETUP/control.py" --phase "$EXTENSION_PHASE" --status running
python "$EXTENSION_SETUP/source/report_math_length_extension.py" --setup "$EXTENSION_SETUP" --models sft \
  --reference-length-report runs/deepseek_length2048_20261015/report/comparison.json \
  > "$EXTENSION_SETUP/report.log" 2>&1
python "$EXTENSION_SETUP/control.py" --phase finished --status complete_verified --exit-code 0
