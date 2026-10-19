"""Resume-compatible budgets, effective-group accounting and bounded retention."""
import json,os,shutil,time
from pathlib import Path
import torch


def validate_extension(old,new,completed):
    assert 0<completed<=old['rounds']<new['rounds']
    allowed={'rounds','save_every','validation_every','script_sha256','longrun_support_sha256','checkpoint_retention'}
    assert {k:v for k,v in old.items() if k not in allowed}=={k:v for k,v in new.items() if k not in allowed}, 'Policy/data settings changed during extension'
    assert new['save_every']>0 and new['validation_every']>0


def seed_history(source,output,state):
    """Copy (never hardlink append-only logs) and audit all existing group totals."""
    with (source/'training.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
    assert len(rows)==state['completed_rounds']
    assert [r['round'] for r in rows]==list(range(1,state['completed_rounds']+1))
    totals={'effective_groups_found':0,'effective_groups_used':0,'all_wrong_groups':0,'all_correct_groups':0}
    for row in rows:
        totals['effective_groups_found']+=row['counts']['mixed']
        totals['effective_groups_used']+=state['config']['groups']
        totals['all_wrong_groups']+=row['counts'].get('all_wrong',0)
        totals['all_correct_groups']+=row['counts'].get('all_correct',0)
    assert sum(row['candidate_groups'] for row in rows)==state['candidate_groups']
    assert totals['effective_groups_used']*state['config']['samples']==state['accepted_responses']
    assert totals['effective_groups_found']+totals['all_wrong_groups']+totals['all_correct_groups']==state['candidate_groups']
    assert sum(row['response_tokens'] for row in rows)==state['trained_response_tokens']
    state.update(totals)
    for name in ('training.jsonl','rollouts.jsonl'):
        assert not (output/name).exists();shutil.copy2(source/name,output/name)
    for path in source.glob('validation_round*.jsonl'):shutil.copy2(path,output/path.name)
    if (source/'validation_before.json').exists():shutil.copy2(source/'validation_before.json',output/'validation_before.json')
    # Preserve the historical best without ever modifying its source files.
    if (source/'best_checkpoint').exists():shutil.copytree(source/'best_checkpoint',output/'best_checkpoint',copy_function=os.link)
    assert not (output/'latest_checkpoint').exists()
    shutil.copytree(source/'final_checkpoint',output/'latest_checkpoint',copy_function=os.link)
    return totals


def add_group_totals(state,counts,used):
    state['effective_groups_found']=state.get('effective_groups_found',0)+counts.get('mixed',0)
    state['effective_groups_used']=state.get('effective_groups_used',0)+used
    state['all_wrong_groups']=state.get('all_wrong_groups',0)+counts.get('all_wrong',0)
    state['all_correct_groups']=state.get('all_correct_groups',0)+counts.get('all_correct',0)
    assert state['effective_groups_used']*state['config']['samples']==state['accepted_responses']
    assert state['effective_groups_found']>=state['effective_groups_used']


def retain_weights(full_checkpoint,destination,state):
    """Immutable FP32 weights every interval; latest/best retain full Adam state."""
    temporary=destination.with_name(destination.name+'.incomplete')
    assert not destination.exists() and not temporary.exists();temporary.mkdir(parents=True)
    for file in full_checkpoint.iterdir():
        if file.name not in ('training_state.pt','checkpoint_manifest.json') and file.is_file():
            os.link(file,temporary/file.name)
    metadata={**state,'checkpoint_storage':{'format':'FP32 weights + counters + RNG; optimizer moments omitted',
        'exact_optimizer_resume':False,'full_resume_location':'latest_checkpoint/training_state.pt'}}
    (temporary/'checkpoint_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    torch.save({'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'completed_rounds':state['completed_rounds'],'training_order_cursor':state['training_order_cursor']},temporary/'rng_state.pt')
    temporary.rename(destination)
    return {'round':state['completed_rounds'],'path':str(destination),'effective_groups_found':state['effective_groups_found'],
            'effective_groups_used':state['effective_groups_used'],'gradient_updates':state['gradient_updates']}
