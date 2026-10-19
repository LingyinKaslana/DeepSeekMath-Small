import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

run = Path(__file__).resolve().parent
root = run.parent.parent

def read(path):
    return json.loads(path.read_text())

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

manifest = read(run / 'manifest.json')
protocol = manifest['protocol']
generation = read(run / 'generation_result.json')
scores = read(run / 'scores.json')
assert generation['status'] == 'generation_complete' and generation['questions'] == 6319
assert scores['status'] == 'complete'
assert protocol['smoke_limit_per_dataset'] is None
assert protocol['trained_input_tokens'] == 0
assert protocol['batch_size'] == 256 and protocol['max_new_tokens'] == 1024
checkpoint = Path(protocol['checkpoint'])
assert checkpoint == root / '.cache/huggingface/hub/models--deepseek-ai--deepseek-coder-1.3b-base/snapshots/c919139c3a9b4070729c8b2cca4847ab29ca8d94'
assert protocol['checkpoint_manifest_sha256'] is None
assert protocol['model_kind'] == 'base'
assert protocol['reference_comparison']['same_generation_settings']
assert protocol['reference_comparison']['same_tokenizer']
assert protocol['reference_comparison']['same_runtime_precision']
assert protocol['model_shards_sha256'] == {'pytorch_model.bin': 'a400155e495b4d6b1cd5b05f04c56032b7dbc6bdff940e046a89ffc1cc2fbdcd'}
assert all(digest(checkpoint / name) == sha for name, sha in protocol['model_shards_sha256'].items())
assert all(digest(Path(path)) == sha for path, sha in protocol['sources_sha256'].items())
for name, count in {'gsm8k': 1319, 'math': 5000}.items():
    predictions = [json.loads(line) for line in (run / f'{name}_predictions.jsonl').read_text().splitlines()]
    scored = [json.loads(line) for line in (run / f'{name}_scored.jsonl').read_text().splitlines()]
    prompts = {item['dataset_index']: item for item in
               (json.loads(line) for line in (run / f'{name}_prompts.jsonl').read_text().splitlines())}
    assert len(predictions) == len(scored) == len(prompts) == count
    assert sorted(item['dataset_index'] for item in predictions) == list(range(count))
    assert sorted(item['dataset_index'] for item in scored) == list(range(count))
    by_index = {item['dataset_index']: item for item in predictions}
    for item in scored:
        index = item['dataset_index']
        assert item['question'] == by_index[index]['question'] == prompts[index]['question']
        assert item['completion'] == by_index[index]['completion']
        assert not item['missing_generation']
    metrics = scores['benchmarks'][name]
    assert metrics['total'] == count and metrics['correct'] == sum(bool(item['correct']) for item in scored)
    assert metrics['missing_generations'] == 0
    assert metrics['prediction_sha256'] == digest(run / f'{name}_predictions.jsonl')
    assert generation['benchmarks'][name]['prediction_sha256'] == metrics['prediction_sha256']
    assert sum(generation['benchmarks'][name]['finish_reasons'].values()) == count
result = {'status': 'complete_verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
          'started_utc': manifest['started_utc'], 'completed_utc': generation['finished_utc'],
          'checkpoint': str(checkpoint), 'model': 'deepseek-ai/deepseek-coder-1.3b-base',
          'trained_input_tokens': 0, 'questions': 6319,
          'protocol': protocol, 'benchmarks': scores['benchmarks'],
          'generation': generation, 'training_modified': False,
          'verification': 'All official test rows generated/scored exactly once; model and protocol hashes unchanged.'}
(run / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
with (run / 'benchmark_scores.csv').open('w', newline='') as stream:
    fields = ['benchmark', 'correct', 'total', 'accuracy_percent', 'truncated_generations', 'timeouts', 'errors']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, metrics in scores['benchmarks'].items():
        writer.writerow({'benchmark': name, **{key: metrics[key] for key in fields[1:]}})
print(json.dumps({'status': result['status'], 'benchmarks':
      {name: {key: metrics[key] for key in ('correct', 'total', 'accuracy_percent', 'truncated_generations', 'timeouts', 'errors')}
       for name, metrics in scores['benchmarks'].items()}}, ensure_ascii=False), flush=True)
