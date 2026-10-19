#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
export PYTHONUNBUFFERED=1
DEEPSEEK_EVAL_RUN=./runs/deepseek_coder_base_inference_20261015
DEEPSEEK_EVAL_STAGE=generation
exec > "$DEEPSEEK_EVAL_RUN/full_inference.log" 2>&1
on_exit() {
    DEEPSEEK_EVAL_EXIT=$?
    trap - EXIT
    python - "$DEEPSEEK_EVAL_RUN" "$DEEPSEEK_EVAL_STAGE" "$DEEPSEEK_EVAL_EXIT" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
run=Path(sys.argv[1]);stage=sys.argv[2];code=int(sys.argv[3])
record={'status':'complete_verified' if code==0 and stage=='finished' else 'failed',
        'stage':stage,'exit_code':code,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'run':str(run),'model':'Original DeepSeek-Coder-1.3B-Base before our CPT',
        'benchmarks':{'gsm8k':1319,'math':5000},'training_modified':False}
if (run/'result.json').exists():record['result']=str(run/'result.json')
for path in (run/'status.json',run.parent/'deepseek_math_inference_active_run.json'):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
PY
    exit "$DEEPSEEK_EVAL_EXIT"
}
trap on_exit EXIT
python "$DEEPSEEK_EVAL_RUN/source/run_math_inference.py" \
 --checkpoint .cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-base/snapshots/c919139c3a9b4070729c8b2cca4847ab29ca8d94 \
 --model-kind base --reference-run runs/deepseek_coder_10b_inference_20261015 \
 --vendor runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math \
 --output "$DEEPSEEK_EVAL_RUN" --batch-size 256 --max-new-tokens 1024
DEEPSEEK_EVAL_STAGE=scoring
python "$DEEPSEEK_EVAL_RUN/source/score_math_benchmarks.py" \
 --run "$DEEPSEEK_EVAL_RUN" --vendor runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math
DEEPSEEK_EVAL_STAGE=verification
python "$DEEPSEEK_EVAL_RUN/finalize.py"
DEEPSEEK_EVAL_STAGE=comparison
.venv-plot/bin/python "$DEEPSEEK_EVAL_RUN/source/compare_math_inference.py" \
 --base-run "$DEEPSEEK_EVAL_RUN" --cpt-run runs/deepseek_coder_10b_inference_20261015 \
 --output runs/deepseek_base_vs_10b_20261015
DEEPSEEK_EVAL_STAGE=finished
