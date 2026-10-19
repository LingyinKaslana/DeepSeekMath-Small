#!/usr/bin/env bash
set -euo pipefail
CPT_PROJECT=.
CPT_INFERENCE_RUN="$CPT_PROJECT/runs/cpt_10b_inference_20261012"
cd "$CPT_PROJECT"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
exec > "$CPT_INFERENCE_RUN/full_inference.log" 2>&1
CPT_INFERENCE_STAGE=generation
on_exit() {
    CPT_INFERENCE_EXIT=$?
    trap - EXIT
    python - "$CPT_INFERENCE_RUN" "$CPT_INFERENCE_STAGE" "$CPT_INFERENCE_EXIT" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
run=Path(sys.argv[1]);stage=sys.argv[2];code=int(sys.argv[3])
record={'status':'complete' if code==0 and stage=='finished' else 'failed','stage':stage,'exit_code':code,'updated_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'model':'final 10B CPT checkpoint','benchmarks':{'gsm8k':1319,'math':5000},'training_modified':False}
if (run/'generation_result.json').exists(): record['generation']=json.loads((run/'generation_result.json').read_text())
if (run/'scores.json').exists(): record['scores']=str(run/'scores.json')
(run/'status.json').write_text(json.dumps(record,indent=2)+'\n')
PY
    exit "$CPT_INFERENCE_EXIT"
}
trap on_exit EXIT
python "$CPT_INFERENCE_RUN/source/run_math_inference.py" \
    --checkpoint "$CPT_PROJECT/runs/cpt_10b_decay_5b_6b_20261010/final_checkpoint" \
    --vendor "$CPT_INFERENCE_RUN/vendor/DeepSeek-Math" \
    --output "$CPT_INFERENCE_RUN" --batch-size 512
CPT_INFERENCE_STAGE=scoring
python "$CPT_INFERENCE_RUN/source/score_math_benchmarks.py" \
    --run "$CPT_INFERENCE_RUN" --vendor "$CPT_INFERENCE_RUN/vendor/DeepSeek-Math"
CPT_INFERENCE_STAGE=finished
