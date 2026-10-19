"""CPU checks for complete native migration without changing CPT math/data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from packed_data import PackedTokens
from source_migration import (apply_native_migration, authorize_native_signature, authorize_signature,
                              validate_lineage, validate_native_backend_change, verify_lineage_snapshots)
from standard_native import (adapt_native_optimizer, make_native_optimizer, prepare_train_probe,
                             preserve_rng, set_native_lr, update_native)
import train_cpt
from verify_cpt_checkpoint import verify_optimizer


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.bad_gradient = False

    def forward(self, input_ids, labels, attention_mask, **kwargs):
        result = self.linear(input_ids).float().square().mean()
        if self.bad_gradient:
            self.linear.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
        return SimpleNamespace(loss=result)


class NativeMigrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.old = json.loads((ROOT / 'configs/cpt_backend_fast.json').read_text())
        self.native = json.loads((ROOT / 'configs/cpt_backend_native.json').read_text())

    def test_actual_backend_changes_are_narrow_and_metadata_classified(self):
        changes = validate_native_backend_change(self.old, self.native)
        fields = {entry['field'] for entry in changes}
        self.assertEqual(fields, {'execution_mode', 'mask_policy', 'compiled_optimizer',
                                 'options.compile_scope', 'purpose', 'validation_evidence'})
        self.assertEqual({entry['field'] for entry in changes if entry['kind'] == 'description_only'},
                         {'purpose', 'validation_evidence'})
        config = train_cpt.load_backend(ROOT / 'configs/cpt_backend_native.json')
        self.assertEqual(config['execution_mode'], 'standard_native')

    def test_native_flag_never_changes_math_data_or_other_kernels(self):
        for key, value in (('sequence_length', 1024), ('micro_batch', 8), ('effective_batch', 32),
                           ('revision', 'bad'), ('fused_optimizer', True), ('unknown', None)):
            bad = copy.deepcopy(self.native)
            bad[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_native_backend_change(self.old, bad)
        for key, value in (('emulate_precision_casts', False), ('loss', 'liger_ce_fp32'),
                           ('kernels', 'rms'), ('preserve_silu', True)):
            bad = copy.deepcopy(self.native)
            bad['options'][key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_native_backend_change(self.old, bad)
        original = {'backend_config_sha256': 'a' * 64, 'source_sha256': {'old.py': 'c' * 64},
                    'peak_lr': 1e-5, 'seed': 42, 'epochs': 2, 'sequence_length': 2048,
                    'effective_batch': 16, 'data_manifest_sha256': 'd' * 64}
        current = dict(original, backend_config_sha256='b' * 64, source_sha256={'new.py': 'e' * 64})
        self.assertTrue(authorize_native_signature(original, current, self.old, self.native))
        with self.assertRaises(ValueError):
            authorize_signature(original, current, True)
        for key in ('peak_lr', 'seed', 'epochs', 'sequence_length', 'effective_batch', 'data_manifest_sha256'):
            bad = dict(current)
            bad[key] = 'changed'
            with self.assertRaises(ValueError, msg=key):
                authorize_native_signature(original, bad, self.old, self.native)
        with self.assertRaises(ValueError):
            validate_native_backend_change(self.native, self.old)

    @unittest.skipUnless((ROOT / 'runs/cpt_10b_20261009_after_native_compare/stopped_checkpoint/training_state.pt').exists(), 'Requires local checkpoint; excluded from release')
    def test_real_checkpoint_native_migration_lineage_and_snapshot_hashes(self):
        checkpoint = ROOT / 'runs/cpt_10b_20261009_after_native_compare/stopped_checkpoint'
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', mmap=True, weights_only=True)
        signature = dict(state['signature'], backend_config_sha256=sha(ROOT / 'configs/cpt_backend_native.json'),
                         source_sha256={name: sha(ROOT / 'scripts' / name) for name in train_cpt.SOURCE_FILES})
        data = PackedTokens(ROOT / 'data/processed/openwebmath_5b_s42', include_tail=True, pad_token_id=151643)
        args = argparse.Namespace(allow_native_backend_migration=True, allow_source_migration=False,
                    resume=checkpoint, backend_config=ROOT / 'configs/cpt_backend_native.json',
                    epochs=2, seed=42, steps=None, lr=1e-5)
        orders = train_cpt.load_saved_orders(state, checkpoint)
        train_cpt.validate_resume(state, signature, data, args, orders=orders)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'source').mkdir()
            for name in train_cpt.SOURCE_FILES:
                shutil.copyfile(ROOT / 'scripts' / name, output / 'source' / name)
            shutil.copyfile(ROOT / 'configs/cpt_backend_native.json', output / 'backend_config.json')
            # Only metadata is needed for the lineage; do not copy 12GB Adam tensors.
            branch = {key: value for key, value in state.items() if key not in ('optimizer', 'cpu_rng', 'cuda_rng')}
            old_steps = branch['completed_steps']
            old_counters = tuple(branch[key] for key in ('optimizer_updates', 'trained_tokens', 'supervised_tokens'))
            apply_native_migration(branch, signature, checkpoint, output, sha)
            result = verify_lineage_snapshots(branch, sha)
            self.assertEqual(result['native_backend_migration_count'], 1)
            self.assertEqual(branch['completed_steps'], old_steps)
            self.assertEqual(tuple(branch[key] for key in ('optimizer_updates', 'trained_tokens', 'supervised_tokens')), old_counters)
            bad = copy.deepcopy(branch)
            bad['source_migrations'][-1]['backend_changes'][0]['after'] = 'tampered'
            with self.assertRaises(ValueError):
                validate_lineage(bad)
            (output / 'backend_config.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_lineage_snapshots(branch, sha)

    @unittest.skipUnless((ROOT / 'data/processed/openwebmath_5b_s42/train_tokens.npy').exists(), 'Requires local token arrays; excluded from release')
    def test_actual_probe_matches_prior_ab_hash_and_rng_isolation(self):
        data = PackedTokens(ROOT / 'data/processed/openwebmath_5b_s42', include_tail=True, pad_token_id=151643)
        before = torch.get_rng_state().clone()
        probe = prepare_train_probe(data, 256, 20261009)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(probe['indices_sha256'], '0275fe980c1fd3bf853fa63d40c27291d4368b0eb41b4c183127d9f964169a81')
        self.assertEqual(probe['tokens_sha256'], 'f2d447c272813ee27c047b0a895234932d1d05cb1e4275b11b982dfedee42619')
        self.assertEqual(prepare_train_probe(data, 256, 20261009, probe), probe)
        self.assertLess(max(probe['indices']), len(data.tokens) // data.sequence_length)
        with self.assertRaises(ValueError):
            prepare_train_probe(data, 256, 73, probe)
        with preserve_rng(torch.device('cpu')):
            torch.rand(43)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_native_update_ordinary_step_and_failed_gradient_zero_updates(self):
        torch.manual_seed(928)
        model = ToyCausalLM()
        optimizer = make_native_optimizer(model, 1e-5)
        def provider(dataset, indices):
            x = torch.ones((len(indices), 3))
            return {'input_ids': x, 'labels': x.clone(), 'attention_mask': torch.ones_like(x, dtype=torch.int64)}, len(indices) * 2, len(indices) * 3
        loss, norm, inputs, targets, applied = update_native(model, optimizer, None, provider, [2, 9, 3], 2)
        self.assertEqual((inputs, targets, applied), (9, 6, True))
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(norm)))
        for state in optimizer.state.values():
            self.assertEqual(float(state['step']), 1)
            self.assertEqual(state['step'].device.type, 'cpu')
        weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved = copy.deepcopy(optimizer.state_dict())
        model.bad_gradient = True
        with self.assertRaisesRegex(RuntimeError, 'optimizer not applied'):
            update_native(model, optimizer, None, provider, [2, 9, 3], 2)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))
        for index in saved['state']:
            for key, value in saved['state'][index].items():
                self.assertTrue(torch.equal(value, optimizer.state_dict()['state'][index][key]))

    def test_optimizer_adaptation_preserves_values_and_cpu_verifier_precision(self):
        model = ToyCausalLM()
        optimizer = make_native_optimizer(model, 1e-5)
        loss = model(torch.ones(2, 3), None, torch.ones(2, 3)).loss
        loss.backward()
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        saved['param_groups'][0].update(capturable=True, lr=torch.tensor(1e-5))
        audit = adapt_native_optimizer(saved)
        self.assertTrue(audit['tensor_values_unchanged'])
        self.assertEqual(audit['before_tensor_sha256'], audit['after_tensor_sha256'])
        optimizer.load_state_dict(saved)
        self.assertFalse(optimizer.param_groups[0]['capturable'])
        schedule = train_cpt.make_schedule(1, 1e-5)
        state = {'optimizer': optimizer.state_dict(), 'schedule': schedule}
        counters = {'optimizer_updates': 1, 'completed_steps': 1, 'execution_mode': 'standard_native', 'compiled_optimizer': False}
        self.assertTrue(verify_optimizer(state, model, counters)['cpu_adamw_load_state_dict_passed'])
        state['optimizer']['param_groups'][0]['lr'] = 1e-5 # differs from required exact FP32 scalar value
        with self.assertRaises(ValueError):
            verify_optimizer(state, model, counters)
        state['optimizer']['param_groups'][0]['lr'] = float(np.float32(1e-5))
        state['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            verify_optimizer(state, model, counters)
        set_native_lr(optimizer, 3.16e-6)
        self.assertEqual(optimizer.param_groups[0]['lr'], float(np.float32(3.16e-6)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
