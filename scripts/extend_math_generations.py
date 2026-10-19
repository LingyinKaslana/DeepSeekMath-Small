"""Regenerate only length-limited answers; retain other original answers exactly."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from math_benchmark_protocol import load_benchmarks, stop_strings
from run_math_inference import digest, write_json

COUNTS = {'gsm8k': 1319, 'math': 5000}


def indexed(path, count):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == count
    assert sorted(row['dataset_index'] for row in rows) == list(range(count))
    return {row['dataset_index']: row for row in rows}


def selected_indices(rows):
    return sorted(index for index, row in rows.items() if row['finish_reason'] == 'length')


def merge_predictions(old, retries):
    selected = set(selected_indices(old))
    assert set(retries) == selected, 'Retry index set must equal all original length-limited rows'
    merged = []
    for index in sorted(old):
        row = retries[index] if index in selected else old[index]
        assert all(row[key] == old[index][key] for key in
                   ('benchmark', 'dataset_index', 'dataset_id', 'question', 'prompt_tokens'))
        merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.batch_size == 128
    source, vendor, output = (path.resolve() for path in (args.source_run, args.vendor, args.output))
    assert source != output
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    source_result = json.loads((source / 'result.json').read_text())
    protocol = dict(source_manifest['protocol'])
    assert source_result['status'] == 'complete_verified' and source_result['questions'] == 6319
    assert protocol['max_new_tokens'] == 1024 and protocol['batch_size'] == 256
    assert protocol['benchmarks'] == COUNTS and protocol['smoke_limit_per_dataset'] is None
    assert protocol['do_sample'] is False and protocol['n'] == protocol['num_beams'] == 1
    assert version('torch') == protocol['torch'] and version('transformers') == protocol['transformers']
    for path, sha in protocol['sources_sha256'].items():
        assert digest(Path(path)) == sha, f'Original protocol source changed: {path}'
    checkpoint = Path(protocol['checkpoint'])
    for name, sha in protocol['model_shards_sha256'].items():
        assert digest(checkpoint / name) == sha, f'Checkpoint weights changed: {name}'
    if protocol['checkpoint_manifest_sha256'] is not None:
        assert digest(checkpoint / 'checkpoint_manifest.json') == protocol['checkpoint_manifest_sha256']
    datasets = load_benchmarks(vendor)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    config = json.loads((checkpoint / 'config.json').read_text())
    original, prompts, selected, evidence = {}, {}, {}, {}
    for name, count in COUNTS.items():
        old = indexed(source / f'{name}_predictions.jsonl', count)
        prompt = indexed(source / f'{name}_prompts.jsonl', count)
        indices = selected_indices(old)
        for index, row in old.items():
            gold = datasets[name][index]
            assert row['question'] == prompt[index]['question'] == gold.get('question', gold.get('problem'))
            assert row['dataset_id'] == prompt[index]['dataset_id'] == gold['id']
        token_ids = tokenizer([prompt[index]['prompt'] for index in indices], add_special_tokens=False)['input_ids']
        assert all(len(ids) == prompt[index]['prompt_tokens'] for index, ids in zip(indices, token_ids))
        assert max(map(len, token_ids)) + 2048 <= config['max_position_embeddings']
        shutil.copyfile(source / f'{name}_prompts.jsonl', output / f'{name}_prompts.jsonl')
        original[name], prompts[name], selected[name] = old, prompt, indices
        evidence[name] = {'selected_indices': indices, 'selected_questions': len(indices),
                          'retained_questions': count - len(indices),
                          'original_predictions_sha256': digest(source / f'{name}_predictions.jsonl'),
                          'original_prompts_sha256': digest(source / f'{name}_prompts.jsonl')}
    protocol.update(max_new_tokens=2048, batch_size=128,
                    evaluation_mode='selective_length_extension',
                    length_extension={'source_run': str(source), 'original_max_new_tokens': 1024,
                                      'original_batch_size': 256, 'selection': "finish_reason == 'length' only",
                                      'source_manifest_sha256': digest(source / 'manifest.json'),
                                      'source_result_sha256': digest(source / 'result.json'),
                                      'benchmarks': evidence,
                                      'regeneration': 'Original prompt, not continuation of saved text',
                                      'retained': 'Non-length-limited original prediction dictionaries copied unchanged'},
                    limits=['Selective regeneration, not an independent full-set 2048-token rerun.',
                            'Retry batches differ from original batches; greedy outputs need not be bitwise identical.',
                            'Prefix equality is recorded for every regenerated answer.',
                            'All official test questions remain in the score denominator.'])
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
                'extension_script_sha256': digest(Path(__file__))}
    if args.resume:
        previous = json.loads((output / 'manifest.json').read_text())
        assert previous['protocol'] == protocol and previous['extension_script_sha256'] == manifest['extension_script_sha256']
        manifest = previous
    else:
        assert not any(output.glob('*retry_predictions.jsonl')), 'Use --resume for an existing retry run'
        write_json(output / 'manifest.json', manifest)
    print(json.dumps({'event': 'prepared', 'source': str(source),
                      'selected': {key: len(value) for key, value in selected.items()},
                      'max_new_tokens': 2048, 'batch_size': 128, 'prepare_only': args.prepare_only}), flush=True)
    if args.prepare_only:
        return
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval()
    model.config.use_cache = True
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    print(json.dumps({'event': 'model_loaded', 'seconds': time.perf_counter()-start}), flush=True)
    total = sum(map(len, selected.values()))
    finished = 0
    summaries = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name in COUNTS:
            destination = output / f'{name}_retry_predictions.jsonl'
            retries = {}
            if args.resume and destination.exists():
                with destination.open() as retry_stream:
                    retry_lines = list(retry_stream)
                for line in retry_lines:
                    row = json.loads(line)
                    assert row['dataset_index'] in selected[name] and row['dataset_index'] not in retries
                    retries[row['dataset_index']] = row
            finished += len(retries)
            remaining = sorted((prompts[name][index] for index in selected[name] if index not in retries),
                               key=lambda item: (item['prompt_tokens'], item['dataset_index']))
            generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=2048,
                use_cache=True, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, stop_strings=stop_strings(name))
            with destination.open('a') as stream:
                for offset in range(0, len(remaining), args.batch_size):
                    batch = remaining[offset:offset+args.batch_size]
                    inputs = tokenizer([item['prompt'] for item in batch], padding=True,
                                       add_special_tokens=False, return_tensors='pt').to('cuda')
                    width = inputs['input_ids'].shape[1]
                    torch.cuda.synchronize()
                    batch_start = time.perf_counter()
                    sequences = model.generate(**inputs, generation_config=generation,
                                               tokenizer=tokenizer, logits_to_keep=1)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-batch_start
                    tokens = sequences[:, width:].cpu().tolist()
                    decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                    prefixes = tokenizer.batch_decode([ids[:1024] for ids in tokens], skip_special_tokens=True)
                    for item, ids, raw, prefix in zip(batch, tokens, decoded, prefixes):
                        eos = next((i for i, token in enumerate(ids) if token == tokenizer.eos_token_id), None)
                        count = len(ids) if eos is None else eos+1
                        stops = [raw.find(stop) for stop in stop_strings(name) if stop in raw]
                        if stops:
                            completion, finish = raw[:min(stops)], 'stop_string'
                            if eos is not None:
                                count -= 1
                        elif eos is not None:
                            completion, finish = raw, 'eos'
                        else:
                            assert len(ids) == 2048
                            completion, finish = raw, 'length'
                        old = original[name][item['dataset_index']]
                        row = {key: old[key] for key in ('benchmark','dataset_index','dataset_id','question','prompt_tokens')}
                        row.update(completion=completion, raw_completion=raw, finish_reason=finish,
                                   generated_tokens=count, generation_batch_seconds=seconds,
                                   generation_batch_size=len(batch), length_extension={
                                       'original_finish_reason': 'length', 'original_max_new_tokens': 1024,
                                       'max_new_tokens': 2048, 'prefix_1024_text_identical': prefix == old['raw_completion']})
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        retries[item['dataset_index']] = row
                    stream.flush()
                    os.fsync(stream.fileno())
                    finished += len(batch)
                    progress = {'status': 'generating', 'updated_utc': datetime.now(timezone.utc).isoformat(),
                                'benchmark': name, 'finished': finished, 'total': total,
                                'dataset_finished': len(retries), 'dataset_total': len(selected[name]),
                                'batch_seconds': seconds, 'max_new_tokens': 2048,
                                'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
                    write_json(output/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                    del inputs, sequences
            merged = merge_predictions(original[name], retries)
            temporary = output / f'{name}_predictions.jsonl.tmp'
            with temporary.open('w') as stream:
                for row in merged:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            temporary.replace(output/f'{name}_predictions.jsonl')
            summaries[name] = {'questions': len(merged), 'rerun_questions': len(retries),
                'retained_questions': COUNTS[name]-len(retries),
                'retry_finish_reasons': dict(Counter(row['finish_reason'] for row in retries.values())),
                'prefix_1024_text_identical': sum(row['length_extension']['prefix_1024_text_identical'] for row in retries.values()),
                'prediction_sha256': digest(output/f'{name}_predictions.jsonl')}
    result = {'status': 'generation_complete', 'questions': 6319, 'rerun_questions': total,
              'benchmarks': summaries, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'wall_seconds': time.perf_counter()-start, 'training_modified': False}
    write_json(output/'generation_result.json', result)
    write_json(output/'progress.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
