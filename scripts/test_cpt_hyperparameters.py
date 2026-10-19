"""Small CPU controls for intervention values, immutable Adam tensors, probe splits."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import compare_cpt_hyperparameters as subject


class Controls(unittest.TestCase):
    def test_lr_multiplier_preserves_schedule_and_fp32_rounding(self):
        schedule = {'peak_lr': 1e-5, 'warmup_steps': 15259}
        original = copy.deepcopy(schedule)
        def lr(s, step):
            return s['peak_lr'] * min((step + 1) / s['warmup_steps'], 1.)
        for step in (0, 4999, 15258, 19999, 20999):
            for case, (factor, _) in subject.CASES.items():
                scheduled, scaled, actual = subject.effective_lr(schedule, step, case, lr)
                self.assertEqual(scheduled, lr(schedule, step))
                self.assertEqual(scaled, lr(schedule, step) * factor)
                self.assertEqual(actual, float(np.float32(scaled)))
        self.assertEqual(schedule, original)

    def test_wd_changes_group_only_and_leaves_all_adam_values_exact(self):
        for case, (_, wd) in subject.CASES.items():
            state = {0: {'step': torch.tensor(20000., dtype=torch.float32),
                         'exp_avg': torch.tensor([.123, -.456]),
                         'exp_avg_sq': torch.tensor([.001, .008])}}
            saved = {'state': state, 'param_groups': [{'params': [0], 'lr': 1e-5,
                'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': .1, 'maximize': False,
                'capturable': False, 'foreach': True, 'fused': False}]}
            before_values = {key: value.clone() for key, value in state[0].items()}
            before_group = copy.deepcopy(saved['param_groups'][0])
            audit = subject.apply_case(saved, case)
            self.assertTrue(audit['adam_tensor_storage_identity_and_versions_unchanged'])
            for key, value in before_values.items():
                self.assertTrue(torch.equal(value, state[0][key]))
            before_group['weight_decay'] = wd
            self.assertEqual(saved['param_groups'][0], before_group)

    def test_seen_unseen_disjoint_and_future_updates_excluded(self):
        orders = SimpleNamespace(epoch_steps=6, orders=[np.arange(12, dtype=np.int32)])
        good = {'start_step': 2, 'end_step': 4, 'groups': {
            'uniform_train': {'indices': [1, 3, 9]},
            'seen_train': {'indices': [0, 2]}, 'unseen_train': {'indices': [8, 11]}}}
        selected = subject.validate_probes(good, orders, 2, 4, 2, 12)
        self.assertFalse(np.intersect1d(selected['seen_train'], selected['unseen_train']).size)
        for name, values in [('unseen_train', [6, 11]), ('seen_train', [0, 5]),
                             ('unseen_train', [2, 11]), ('uniform_train', [1, 1])]:
            bad = copy.deepcopy(good)
            bad['groups'][name]['indices'] = values
            with self.assertRaises(ValueError):
                subject.validate_probes(bad, orders, 2, 4, 2, 12)
        extended = copy.deepcopy(good)
        extended['unseen_exclusion_through_step'] = 5
        with self.assertRaises(ValueError):
            subject.validate_probes(extended, orders, 2, 4, 2, 12)
        extended['groups']['unseen_train']['indices'] = [10, 11]
        subject.validate_probes(extended, orders, 2, 4, 2, 12)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    out = Path('runs/cpt_causes_20261009/hyperparameter_cpu_tests.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'status': 'passed' if result.wasSuccessful() else 'failed',
        'tests_run': result.testsRun, 'gpu_execution': False,
        'failures': len(result.failures), 'errors': len(result.errors),
        'purpose': 'Verify LR-only scaling, WD group-only mutation with exact unchanged Adam values, and disjoint seen/unseen probes excluding all intervention updates'}, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
