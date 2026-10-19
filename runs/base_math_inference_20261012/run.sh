#!/usr/bin/env bash
set -euo pipefail
CPT_PROJECT=.
BASE_INFERENCE_RUN="$CPT_PROJECT/runs/base_math_inference_20261012"
cd "$CPT_PROJECT"
source "$CPT_PROJECT/activate.sh"
export PYTHONUNBUFFERED=1
exec > "$BASE_INFERENCE_RUN/full_inference.log" 2>&1
BASE_INFERENCE_STAGE=generation
on_exit() {
    BASE_INFERENCE_EXIT=$?
    trap - EXIT
    python - "$BASE_INFERENCE_RUN" "$BASE_INFERENCE_STAGE" "$BASE_INFERENCE_EXIT" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
run=Path(sys.argv[1]);stage=sys.argv[2];code=int(sys.argv[3])
record={'status':'complete' if code==0 and stage=='finished' else 'failed','stage':stage,'exit_code':code,'updated_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'model':'pinned original Qwen2.5-Coder-1.5B, before our CPT','benchmarks':{'gsm8k':1319,'math':5000},'training_modified':False}
if (run/'generation_result.json').exists(): record['generation']=json.loads((run/'generation_result.json').read_text())
if (run/'scores.json').exists(): record['scores']=str(run/'scores.json')
(run/'status.json').write_text(json.dumps(record,indent=2)+'\n')
PY
    exit "$BASE_INFERENCE_EXIT"
}
trap on_exit EXIT
python "$BASE_INFERENCE_RUN/source/run_math_inference.py" \
    --checkpoint "$CPT_PROJECT/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B/snapshots/df3ce67c0e24480f20468b6ef2894622d69eb73b" \
    --model-kind base --reference-run "$CPT_PROJECT/runs/cpt_10b_inference_20261012" \
    --vendor "$CPT_PROJECT/runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math" \
    --output "$BASE_INFERENCE_RUN" --batch-size 512 --resume
BASE_INFERENCE_STAGE=scoring
python "$BASE_INFERENCE_RUN/source/score_math_benchmarks.py" \
    --run "$BASE_INFERENCE_RUN" --vendor "$CPT_PROJECT/runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math"
BASE_INFERENCE_STAGE=finished
