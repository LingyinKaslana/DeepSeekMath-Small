"""Freeze the fork and its evaluation samples without touching production."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

ROOT = Path('.')
DIAG = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/cpt_10b_lr3e6_20261009'
START, STEPS = 115000, 6104
END = START + STEPS

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    if (DIAG / 'plan.json').exists():
        raise FileExistsError('Experiment already prepared')
    source = DIAG / 'source'
    source.mkdir()
    for path in (ROOT / 'runs/cpt_10b_lr3e6_20261009_launch/source').glob('*.py'):
        shutil.copy2(path, source / path.name)
    for name in ('compare_cpt_backends.py', 'compare_cpt_hyperparameters.py'):
        shutil.copy2(ROOT / 'runs/cpt_causes_20261009/setup/tools' / name, DIAG / 'tools' / name)
    checkpoint = RUN / 'checkpoint_step_00115000'
    frozen = DIAG / 'starting_checkpoint'
    shutil.copytree(checkpoint, frozen, copy_function=os.link)
    manifest = json.loads((RUN / 'manifest.json').read_text())
    old = json.loads((ROOT / 'runs/cpt_seen_unseen_lr3e6_20261010/probes.json').read_text())
    data = ROOT / 'data/processed/openwebmath_5b_s42'
    order = np.load(frozen / 'orders/order_epoch_00000.npy', mmap_mode='r')
    tokens = np.load(data / 'train_tokens.npy', mmap_mode='r')
    full = len(tokens) // 2048
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order))
    pool = np.asarray(order[END * 16:])
    pool = pool[pool < full]
    unseen = np.sort(np.random.default_rng(20261012).choice(pool, 256, replace=False)).astype(np.int64)

    def group(indices, sampling):
        indices = np.asarray(indices, dtype=np.int64)
        digest = hashlib.sha256()
        for index in indices:
            digest.update(memoryview(np.ascontiguousarray(tokens[int(index)*2048:(int(index)+1)*2048])).cast('B'))
        return dict(indices=indices.tolist(), blocks=len(indices),
            indices_sha256=hashlib.sha256(memoryview(indices).cast('B')).hexdigest(),
            tokens_sha256=digest.hexdigest(), input_tokens=len(indices)*2048,
            supervised_tokens=len(indices)*2047, sampling=sampling,
            epoch0_order_positions=positions[indices].tolist(),
            seen_blocks_at_start=int((positions[indices] < START*16).sum()),
            seen_blocks_at_end=int((positions[indices] < END*16).sum()))

    groups = {
        'uniform_train': group(old['groups']['uniform_train']['indices'], 'Original fixed uniform production train probe'),
        'seen_train': group(old['groups']['seen_train']['indices'], 'Original early-seen probe, all seen by step15000'),
        'unseen_train': group(unseen, 'Fresh fixed sample never consumed through branch end121104; partial tail excluded')}
    assert groups['seen_train']['seen_blocks_at_start'] == 256
    assert groups['unseen_train']['seen_blocks_at_end'] == 0
    probes = dict(schema_version=1, start_step=START, end_step=END,
        unseen_exclusion_through_step=END, probe_seed=20261012, groups=groups,
        private_rng=True, seen_definition='Exact2048-token blocks in these local branches, not entire documents or prior base pretraining')
    (DIAG / 'probes.json').write_text(json.dumps(probes, indent=2)+'\n')
    history = json.loads((RUN / 'validation_history.json').read_text())
    expected = next(row['loss'] for row in history if row['step'] == START)
    plan = dict(status='prepared', created_utc=datetime.now(timezone.utc).isoformat(),
        production_run=str(RUN), production_pid=621138, production_files_modified=False,
        original_checkpoint=str(checkpoint), checkpoint=str(frozen), source=str(source), data=str(data),
        start_step=START, end_step=END, steps=STEPS, tokens_per_update=32768,
        new_tokens_per_arm=STEPS*32768, initial_lr=3e-6, final_decay_lr=3e-7,
        evaluation_offsets=[0,500,1000,2000,3000,4000,5000,6104],
        expected_initial_validation=expected, initial_validation_tolerance=1e-6,
        diagnostic_micro_batch=8, effective_batch=16, evaluation_micro_batch=4,
        allocator_memory_fraction=.42,
        arms=[dict(id='constant', label='constant LR3e-6', run_directory=str(DIAG/'branch_constant'), lr_schedule='constant'),
              dict(id='linear_decay', label='linear LR3e-6 to3e-7', run_directory=str(DIAG/'branch_linear_decay'), lr_schedule='linear across6104updates; first3e-6 last3e-7')],
        checkpoint_manifest_sha256=sha(frozen/'checkpoint_manifest.json'),
        source_sha256=manifest['source_sha256'], probes_sha256=sha(DIAG/'probes.json'),
        shared_implementation_difference='Both arms use original optimized source with micro8 and two gradient accumulation passes instead of production micro16; effective batch remains16. Compile shape8x2048.',
        unmodified=['starting weights and Adam moments/counters','CPU/CUDA RNG at start','next6104batches and token/label values',
            'effective batch16 and sequence2048','FP32 parameters/grads/moments with BF16autocast','betas0.9/0.95 WD0.1 eps1e-8 clip1','native fixed evaluation'],
        main_intervention_policy='No automatic production change; complete controlled comparison first',
        finite_policy='Fail before optimizer mutation; never skip a batch',
        limits=['Two arms differ only in LR schedule relative to their shared implementation',
            'A local response does not establish the unique original cause',
            'LR changes gradient-update magnitude and LR-times-weight-decay together',
            'Unseen denotes exact blocks in these branches, independent of continuing production',
            'Shared GPU contention affects timing but not token budgets'])
    (DIAG/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({key:plan[key] for key in ('start_step','end_step','new_tokens_per_arm','expected_initial_validation','diagnostic_micro_batch')}))

if __name__ == '__main__':
    main()
