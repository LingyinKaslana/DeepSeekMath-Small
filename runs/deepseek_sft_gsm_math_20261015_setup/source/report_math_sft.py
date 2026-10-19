"""CPU SFT learning curves and protocol-matched CPT/SFT benchmark summary."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())


def plot(run,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (run/'training.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    val=read(run/'validation_history.json');probe=read(run/'train_probe_history.json')
    tokens={0:0,**{row['step']:row['cumulative_sft_input_tokens'] for row in rows}}
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for history,label,color in ((val,'Held-out SFT validation','#2563eb'),(probe,'Fixed train probe','#d97706')):
        axes[0].plot([tokens[row['step']]/1e6 for row in history],[row['loss'] for row in history],marker='o',markersize=3,label=label,color=color)
    axes[0].set_ylabel('Answer-only loss');axes[0].legend(frameon=False)
    means=[]
    for offset in range(0,len(rows),50):
        window=rows[offset:offset+50];targets=sum(row['supervised_tokens'] for row in window)
        means.append((window[-1]['cumulative_sft_input_tokens']/1e6,sum(row['loss']*row['supervised_tokens'] for row in window)/targets))
    axes[1].plot([row[0] for row in means],[row[1] for row in means],color='#059669',label='Batch loss, 50-update token-weighted means')
    axes[1].set_ylabel('Training batch loss');axes[1].set_xlabel('Additional SFT input tokens (millions)')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('DeepSeek-Coder-1.3B after 10B CPT: native full-parameter CoT SFT')
    fig.tight_layout()
    for extension in ('png','pdf'):fig.savefig(output/f'sft_learning_curve.{extension}',dpi=180)
    plt.close(fig)
    write={'status':'complete','generated_utc':datetime.now(timezone.utc).isoformat(),'latest_validation':val[-1],
           'best_validation':min(val,key=lambda row:row['loss']),'latest_train_probe':probe[-1],
           'updates':len(rows),'input_tokens':sum(row['input_tokens'] for row in rows),
           'supervised_tokens':sum(row['supervised_tokens'] for row in rows)}
    (output/'training_summary.json').write_text(json.dumps(write,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup',type=Path,required=True)
    p.add_argument('--training-only',action='store_true')
    a=p.parse_args();setup=a.setup.resolve();run=setup.parent/'deepseek_sft_gsm_math_20261015'
    output=setup/'report';output.mkdir(exist_ok=True)
    plot(run,output)
    if a.training_only:return
    evaluation=setup/'evaluation';manifest=read(evaluation/'manifest.json');protocol=manifest['protocol']
    assert protocol['model_kind']=='sft' and protocol['reference_comparison']['same_generation_settings']
    generation=read(evaluation/'generation_result.json');scores=read(evaluation/'scores.json')
    assert generation['questions']==6319 and generation['status']=='generation_complete' and scores['status']=='complete'
    reference=Path(protocol['reference_comparison']['run']);prior=read(reference/'scores.json')
    comparison={}
    for name,count in {'gsm8k':1319,'math':5000}.items():
        with (evaluation/f'{name}_scored.jsonl').open() as stream:scored=[json.loads(line) for line in stream]
        with (evaluation/f'{name}_predictions.jsonl').open() as stream:predictions=[json.loads(line) for line in stream]
        assert len(scored)==len(predictions)==count
        assert sorted(row['dataset_index'] for row in scored)==list(range(count))
        assert sorted(row['dataset_index'] for row in predictions)==list(range(count))
        by_index={row['dataset_index']:row for row in predictions}
        assert all(row['question']==by_index[row['dataset_index']]['question'] and row['completion']==by_index[row['dataset_index']]['completion'] for row in scored)
        assert sum(row['correct'] for row in scored)==scores['benchmarks'][name]['correct']
        assert scores['benchmarks'][name]['missing_generations']==0
        digest=hashlib.sha256((evaluation/f'{name}_predictions.jsonl').read_bytes()).hexdigest()
        assert digest==scores['benchmarks'][name]['prediction_sha256']==generation['benchmarks'][name]['prediction_sha256']
        assert (evaluation/f'{name}_prompts.jsonl').read_bytes()==(reference/f'{name}_prompts.jsonl').read_bytes()
        old,new=prior['benchmarks'][name],scores['benchmarks'][name]
        comparison[name]={'cpt_correct':old['correct'],'sft_correct':new['correct'],'total':count,
            'cpt_accuracy_percent':old['accuracy_percent'],'sft_accuracy_percent':new['accuracy_percent'],
            'delta_percentage_points':new['accuracy_percent']-old['accuracy_percent'],
            'truncated_generations':new['truncated_generations'],'errors':new['errors'],'timeouts':new['timeouts']}
    result={'status':'complete_verified','verified_utc':datetime.now(timezone.utc).isoformat(),
        'training':read(run/'result.json'),'verification_final':read(setup/'verification_final.json'),
        'evaluation_checkpoint':protocol['checkpoint'],'questions':6319,'max_new_tokens':1024,
        'comparison':comparison,'evaluation_mode':'Same base few-shot prompts as CPT; answer-only SFT uses matching Q/A and Problem/Solution formats',
        'cpt_checkpoint_modified':False}
    (evaluation/'result.json').write_text(json.dumps({'status':'complete_verified','questions':6319,
        'protocol':protocol,'benchmarks':scores['benchmarks']},indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (output/'benchmark_comparison.csv').open('w',newline='') as stream:
        rows=[{'benchmark':name,**value} for name,value in comparison.items()]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':result['status'],'comparison':comparison},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
