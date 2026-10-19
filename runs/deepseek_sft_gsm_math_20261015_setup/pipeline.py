import json
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import traceback

setup=Path(__file__).resolve().parent;root=setup.parent.parent
run=root/'runs/deepseek_sft_gsm_math_20261015';source=setup/'source'
evaluation=setup/'evaluation';phase='startup'

def status(state):
    record={'status':state,'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'sft','setup':str(setup),'run':str(run),'progress':str(run/'progress.json'),
        'training_examples':14459,'validation_examples':512,'epochs':3,'total_steps':1356,
        'peak_lr':1e-5,'min_lr':1e-6,'micro_batch':8,'effective_batch':32,
        'report':str(setup/'report/completion.json'),'evaluation':str(evaluation)}
    for path in (setup/'status.json',root/'runs/sft_active_run.json'):
        temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
    print(json.dumps(record,ensure_ascii=False),flush=True)

def execute(arguments,log):
    with (setup/log).open('w') as stream:
        subprocess.run([str(x) for x in arguments],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)

try:
    assert json.loads((setup/'startup_verification.json').read_text())['status']=='passed'
    phase='training';status('running')
    execute([sys.executable,'-u',source/'train_math_sft.py','--checkpoint',root/'runs/deepseek_coder_1p3b_10b_20261012/final_checkpoint',
        '--data',root/'data/processed/sft_gsm_math_4096_s42_20261015','--output',run,
        '--epochs','3','--lr','1e-5','--micro-batch','8','--effective-batch','32'],'training.log')
    training=json.loads((run/'result.json').read_text())
    assert training['status']=='complete' and training['completed_steps']==1356
    assert training['sft_input_tokens']==10866690 and training['sft_supervised_tokens']==7408092
    phase='checkpoint_verification';status('running')
    execute([sys.executable,source/'verify_math_sft.py','--checkpoint',run/'final_checkpoint',
             '--output',setup/'verification_final.json'],'verification_final.log')
    checkpoint=Path(training['evaluation_checkpoint'])
    assert checkpoint.parent==run and checkpoint.name in ('best_checkpoint','final_checkpoint')
    if checkpoint.name=='best_checkpoint':
        execute([sys.executable,source/'verify_math_sft.py','--checkpoint',checkpoint,
                 '--output',setup/'verification_best.json'],'verification_best.log')
    phase='training_curves';status('running')
    execute([root/'.venv-plot/bin/python',source/'report_math_sft.py','--setup',setup,'--training-only'],'plot.log')
    phase='benchmark_generation';status('running')
    execute([sys.executable,'-u',source/'run_math_inference.py','--model-kind','sft','--checkpoint',checkpoint,
        '--reference-run',root/'runs/deepseek_coder_10b_inference_20261015',
        '--vendor',root/'runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math',
        '--output',evaluation,'--batch-size','256','--max-new-tokens','1024'],'evaluation_generation.log')
    phase='benchmark_scoring';status('running')
    execute([sys.executable,source/'score_math_benchmarks.py','--run',evaluation,
        '--vendor',root/'runs/cpt_10b_inference_20261012/vendor/DeepSeek-Math'],'evaluation_scoring.log')
    phase='comparison';status('running')
    execute([root/'.venv-plot/bin/python',source/'report_math_sft.py','--setup',setup],'report.log')
    assert json.loads((setup/'report/completion.json').read_text())['status']=='complete_verified'
    phase='finished';status('complete_verified')
except BaseException:
    status('failed');traceback.print_exc();raise
