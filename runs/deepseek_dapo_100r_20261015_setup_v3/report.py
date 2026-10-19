import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def rows(path):
    with path.open() as stream:return [json.loads(line) for line in stream]

setup=Path(__file__).resolve().parent;root=setup.parent.parent;run=setup.parent/'deepseek_dapo_100r_20261015'
result=read(run/'result.json');assert result['status']=='complete_verified' and not result['diagnostic_smoke']
assert result['completed_rounds']==100 and result['accepted_responses']==25600 and result['gradient_updates']==400
assert result['source_sft_unchanged'] and result['verification']['status']=='passed'
evaluation=setup/'evaluation';protocol=read(evaluation/'manifest.json')['protocol']
generation=read(evaluation/'generation_result.json');scores=read(evaluation/'scores.json')
assert protocol['model_kind']=='rl' and protocol['max_new_tokens']==1024 and protocol['batch_size']==256
assert generation['status']=='generation_complete' and generation['questions']==6319
assert scores['status']=='complete'
checkpoint=Path(protocol['checkpoint']);best=read(checkpoint/'checkpoint_manifest.json')
assert best['completed_rounds']==result['best_round'] and not best['diagnostic_smoke']
assert sha(checkpoint/'checkpoint_manifest.json')==protocol['checkpoint_manifest_sha256']
for name,digest in protocol['model_shards_sha256'].items():assert sha(checkpoint/name)==digest
for name,digest in read(setup/'source_hashes.json').items():assert sha(setup/'source'/name)==digest
comparison={}
for benchmark,total in (('gsm8k',1319),('math',5000)):
    prediction=rows(evaluation/f'{benchmark}_predictions.jsonl');scored=rows(evaluation/f'{benchmark}_scored.jsonl')
    assert len(prediction)==len(scored)==total
    assert sorted(r['dataset_index'] for r in prediction)==list(range(total))
    assert sorted(r['dataset_index'] for r in scored)==list(range(total))
    indexed={r['dataset_index']:r for r in prediction}
    for row in scored:
        assert not row['missing_generation'] and row['completion']==indexed[row['dataset_index']]['completion']
    metrics=scores['benchmarks'][benchmark];assert metrics['total']==total and metrics['missing_generations']==0
    assert sum(r['correct'] for r in scored)==metrics['correct']
    assert sha(evaluation/f'{benchmark}_predictions.jsonl')==metrics['prediction_sha256']==generation['benchmarks'][benchmark]['prediction_sha256']
    sft=read(root/'runs/deepseek_sft_gsm_math_20261015_setup/evaluation/scores.json')['benchmarks'][benchmark]
    comparison[benchmark]={'sft_correct':sft['correct'],'rl_correct':metrics['correct'],'total':total,
        'sft_accuracy_percent':100*sft['correct']/total,'rl_accuracy_percent':100*metrics['correct']/total,
        'delta_percentage_points':100*(metrics['correct']-sft['correct'])/total,
        'truncated':metrics['truncated_generations'],'errors':metrics['errors'],'timeouts':metrics['timeouts']}
verified={'status':'complete_verified','verified_utc':datetime.now(timezone.utc).isoformat(),
    'questions':6319,'checkpoint':str(checkpoint),'max_new_tokens':1024,'benchmarks':scores['benchmarks']}
(evaluation/'result.json').write_text(json.dumps(verified,indent=2)+'\n')
output=setup/'report';output.mkdir(exist_ok=True)
report={'status':'complete_verified','verified_utc':datetime.now(timezone.utc).isoformat(),
    'training':result,'selected_checkpoint':str(checkpoint),'selected_round':result['best_round'],
    'comparison':comparison,'evaluation_protocol':'Full official GSM8K/MATH; original few-shot prompts; greedy1024; batch256',
    'limits':['RL uses GSM8K training questions already seen in SFT.',
        'Selection uses separate 256-question zero-shot GSM8K validation, not official tests.',
        'No semantic or comprehensive CPT benchmark decontamination.',
        'Response count budget about1% of DAPO; token count and FLOPs are not matched.',
        'Single training seed; no statistical stability claim.']}
(output/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
