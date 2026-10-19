"""Verify selective 2048-token retries and compare merged complete scores."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

COUNTS = {'gsm8k': 1319, 'math': 5000}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def indexed(path, count):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == count and sorted(row['dataset_index'] for row in rows) == list(range(count))
    return {row['dataset_index']: row for row in rows}


def accuracy(rows):
    count = len(rows)
    correct = sum(bool(row['correct']) for row in rows)
    return {'correct': correct, 'total': count, 'accuracy_percent': 100*correct/count if count else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=('base','cpt','sft'), default=['base','cpt'])
    parser.add_argument('--reference-length-report', type=Path)
    args = parser.parse_args()
    setup = args.setup.resolve()
    output = setup/'report'
    output.mkdir(exist_ok=True)
    models, transitions, csv_rows = {}, [], []
    assert len(set(args.models)) == len(args.models)
    for model in args.models:
        run = setup/model
        manifest, scores, generation = (read(run/name) for name in
                                       ('manifest.json','scores.json','generation_result.json'))
        protocol = manifest['protocol']
        assert protocol['model_kind'] == model
        extension = protocol['length_extension']
        source = Path(extension['source_run'])
        assert scores['status'] == 'complete' and generation['status'] == 'generation_complete'
        assert generation['questions'] == 6319
        assert protocol['max_new_tokens'] == 2048 and protocol['batch_size'] == 128
        assert protocol['evaluation_mode'] == 'selective_length_extension'
        assert sha(source/'manifest.json') == extension['source_manifest_sha256']
        assert sha(source/'result.json') == extension['source_result_sha256']
        for name, digest in protocol['model_shards_sha256'].items():
            assert sha(Path(protocol['checkpoint'])/name) == digest
        benchmarks = {}
        for benchmark, count in COUNTS.items():
            old = indexed(source/f'{benchmark}_scored.jsonl',count)
            old_pred = indexed(source/f'{benchmark}_predictions.jsonl',count)
            new = indexed(run/f'{benchmark}_scored.jsonl',count)
            new_pred = indexed(run/f'{benchmark}_predictions.jsonl',count)
            evidence = extension['benchmarks'][benchmark]
            selected = set(evidence['selected_indices'])
            assert selected == {i for i,row in old_pred.items() if row['finish_reason']=='length'}
            assert sha(source/f'{benchmark}_predictions.jsonl') == evidence['original_predictions_sha256']
            assert sha(source/f'{benchmark}_prompts.jsonl') == evidence['original_prompts_sha256']
            assert sha(run/f'{benchmark}_prompts.jsonl') == evidence['original_prompts_sha256']
            with (run/f'{benchmark}_retry_predictions.jsonl').open() as stream:
                retry_rows = [json.loads(line) for line in stream]
            assert len(retry_rows) == len(selected) and {r['dataset_index'] for r in retry_rows} == selected
            for index in old:
                assert new[index]['completion'] == new_pred[index]['completion']
                assert new[index]['question'] == old[index]['question']
                assert not new[index]['missing_generation']
                if index not in selected:
                    assert new_pred[index] == old_pred[index], 'Non-truncated prediction was modified'
                    assert new[index]['correct'] == old[index]['correct'], 'Retained answer score changed'
                else:
                    assert new_pred[index]['length_extension']['max_new_tokens'] == 2048
                    transitions.append({'model':model,'benchmark':benchmark,'dataset_index':index,
                        'old_correct':old[index]['correct'],'new_correct':new[index]['correct'],
                        'new_finish_reason':new_pred[index]['finish_reason'],
                        'prefix_1024_text_identical':new_pred[index]['length_extension']['prefix_1024_text_identical']})
            full_old, full_new = accuracy(list(old.values())), accuracy(list(new.values()))
            metrics = scores['benchmarks'][benchmark]
            assert full_new['correct'] == metrics['correct'] and metrics['total'] == count
            assert sha(run/f'{benchmark}_predictions.jsonl') == metrics['prediction_sha256'] == generation['benchmarks'][benchmark]['prediction_sha256']
            assert metrics['missing_generations'] == 0
            selected_rows = [new[i] for i in selected]
            groups = {}
            for same in (True,False):
                indices = [i for i in selected if new_pred[i]['length_extension']['prefix_1024_text_identical']==same]
                groups[str(same)] = {'old':accuracy([old[i] for i in indices]),'new':accuracy([new[i] for i in indices])}
            result = {'original_1024':full_old,'selectively_extended_2048':full_new,
                'delta_percentage_points':full_new['accuracy_percent']-full_old['accuracy_percent'],
                'rerun_questions':len(selected),'retained_questions':count-len(selected),
                'original_truncated_correct':sum(old[i]['correct'] for i in selected),
                'retry_correct':sum(row['correct'] for row in selected_rows),
                'gained':sum(not old[i]['correct'] and new[i]['correct'] for i in selected),
                'lost':sum(old[i]['correct'] and not new[i]['correct'] for i in selected),
                'still_truncated_2048':metrics['truncated_generations'],
                'prefix_identical_questions':sum(new_pred[i]['length_extension']['prefix_1024_text_identical'] for i in selected),
                'prefix_groups':groups,'errors':metrics['errors'],'timeouts':metrics['timeouts']}
            benchmarks[benchmark] = result
            csv_rows.append({'model':model,'benchmark':benchmark,
                'original_1024_correct':full_old['correct'],'selective_2048_correct':full_new['correct'],
                'total':count,'original_1024_accuracy_percent':full_old['accuracy_percent'],
                'selective_2048_accuracy_percent':full_new['accuracy_percent'],
                **{key:result[key] for key in ('delta_percentage_points','rerun_questions','gained','lost','still_truncated_2048','prefix_identical_questions')}})
        model_result = {'status':'complete_verified','verified_utc':datetime.now(timezone.utc).isoformat(),
            'questions':6319,'rerun_questions':generation['rerun_questions'],'source_run':str(source),
            'checkpoint':protocol['checkpoint'],'evaluation_mode':'selective_length_extension',
            'max_new_tokens':2048,'benchmarks':benchmarks,'training_modified':False}
        (run/'result.json').write_text(json.dumps(model_result,ensure_ascii=False,indent=2)+'\n')
        models[model] = model_result
    comparison = {}
    protocols = [read(setup/model/'manifest.json')['protocol'] for model in args.models]
    for other in protocols[1:]:
        for key in ('max_new_tokens','batch_size','prompt_format','do_sample','num_beams','n','seed',
                    'backend','compute_precision','torch','transformers','official_repository_commit'):
            assert protocols[0][key] == other[key], f'Extended protocol mismatch: {key}'
    for benchmark in COUNTS:
        for model in args.models[1:]:
            assert sha(setup/args.models[0]/f'{benchmark}_prompts.jsonl') == sha(setup/model/f'{benchmark}_prompts.jsonl')
        comparison[benchmark]={model:models[model]['benchmarks'][benchmark]['selectively_extended_2048'] for model in args.models}
        if 'base' in models and 'cpt' in models:
            comparison[benchmark]['cpt_minus_base_percentage_points']=(comparison[benchmark]['cpt']['accuracy_percent']-comparison[benchmark]['base']['accuracy_percent'])
    if args.reference_length_report is not None:
        reference=read(args.reference_length_report)
        assert reference['status']=='complete_verified' and reference['evaluation_mode']=='selective_length_extension'
        assert 'sft' in models
        for benchmark in COUNTS:
            baseline=reference['models']['cpt']['benchmarks'][benchmark]['selectively_extended_2048']
            comparison[benchmark]['cpt']=baseline
            comparison[benchmark]['sft_minus_cpt_percentage_points']=(comparison[benchmark]['sft']['accuracy_percent']-baseline['accuracy_percent'])
    result = {'status':'complete_verified','verified_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_mode':'selective_length_extension','models':models,'comparison':comparison,
        'limits':['Only each model\'s own originally length-limited answers were regenerated.',
                  'All other original answers retained exactly. Full denominators are unchanged.',
                  'This is a selectively extended evaluation, not an independent full-set 2048 run.',
                  'Regeneration batch/shape changes can change the first1024 tokens; prefix equality and separate groups are reported.'],
        'training_modified':False}
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for filename,rows in (('benchmark_scores.csv',csv_rows),('retry_transitions.csv',transitions)):
        with (output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':result['status'],'comparison':comparison},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
