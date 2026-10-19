"""Read-only native evaluation on fixed seen/unseen training blocks."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--probes', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--label', required=True)
    p.add_argument('--expected-validation', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source.resolve()))
    import numpy as np
    import torch
    import train_cpt as trainer
    from packed_data import PackedTokens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    probes = json.loads(a.probes.read_text())
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, local_files_only=True)
    train = PackedTokens(a.data, sequence_length=2048, include_tail=True,
                         pad_token_id=tokenizer.eos_token_id)
    validation = PackedTokens(a.data, 'validation', sequence_length=2048)
    train.explicit_mask = validation.explicit_mask = True
    checked = {}
    for name, group in probes['groups'].items():
        indices = np.array(group['indices'], dtype=np.int64)
        assert len(indices) == 256 and len(np.unique(indices)) == 256
        assert int(indices.min()) >= 0 and int(indices.max()) < len(train.tokens) // 2048
        index_sha = hashlib.sha256(memoryview(indices).cast('B')).hexdigest()
        assert index_sha == group['indices_sha256']
        digest = hashlib.sha256()
        for index in indices:
            digest.update(memoryview(np.ascontiguousarray(train.tokens[int(index)*2048:(int(index)+1)*2048])).cast('B'))
        assert digest.hexdigest() == group['tokens_sha256']
        checked[name] = {'indices_sha256': index_sha, 'tokens_sha256': digest.hexdigest()}
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
        local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda')
    model.config.use_cache = False
    metrics = {'validation': trainer.evaluate(model, validation, trainer.cpt.batch, native_standard=True)}
    if abs(metrics['validation']['loss'] - a.expected_validation) > 1e-6:
        raise RuntimeError('Full validation differs from the immutable reference')
    for name, group in probes['groups'].items():
        metrics[name] = trainer.evaluate(model, train, trainer.cpt.batch,
                                        indices=group['indices'], native_standard=True)
        print(json.dumps({'event': 'evaluation', 'label': a.label, 'group': name,
                          'loss': metrics[name]['loss']}), flush=True)
    result = {'status': 'complete', 'label': a.label, 'model': a.model,
              'revision': a.revision, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'probes_sha256': sha(a.probes),
              'data_manifest_sha256': sha(a.data / 'manifest.json'),
              'source_sha256': {name: sha(a.source / name) for name in trainer.SOURCE_FILES},
              'checked_probe_hashes': checked, 'metrics': metrics,
              'seconds': time.perf_counter() - started,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers')},
              'precision': 'uncompiled native HF SDPA, BF16 autocast, FP32 weights',
              'training_updates': 0, 'production_files_modified': False,
              'limit': 'Seen means selected block occurred in this CPT prefix. It does not establish absence of overlap in prior base pretraining or neighboring blocks.'}
    write(a.output / 'result.json', result)
    print(json.dumps({'status': 'complete', 'label': a.label,
                      'metrics': {name: row['loss'] for name, row in metrics.items()}}), flush=True)


if __name__ == '__main__':
    main()
