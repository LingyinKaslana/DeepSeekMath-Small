"""CPU checks for diagnostic-only Adam adaptation, safe updates, and data mapping."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

import compare_cpt_backends as comparison
from cpt_orders import EpochOrders, prefix_counts
from packed_data import PackedTokens
from token_batches import PackedTokenBatches


class DiagnosticTests(unittest.TestCase):
    def toy_state(self):
        torch.manual_seed(5821)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95),
                                     weight_decay=.1, foreach=True, fused=False)
        for _ in range(7):
            optimizer.zero_grad(set_to_none=True)
            loss = model(torch.randn(2, 4)).square().mean()
            loss.backward()
            optimizer.step()
        return model, optimizer

    def test_native_implementation_adaptation_preserves_all_values(self):
        model, optimizer = self.toy_state()
        saved = copy.deepcopy(optimizer.state_dict())
        saved['param_groups'][0].update(lr=torch.tensor(1e-5, dtype=torch.float32), capturable=True)
        before = comparison.state_audit(saved, torch, expected_updates=7)
        before_lr = float(saved['param_groups'][0]['lr'])
        comparison.adapt_native_optimizer(saved, torch)
        after = comparison.state_audit(saved, torch, expected_updates=7)
        self.assertEqual(before['combined_tensor_sha256'], after['combined_tensor_sha256'])
        self.assertEqual(float(saved['param_groups'][0]['lr']), before_lr)
        reference = copy.deepcopy(model)
        native = torch.optim.AdamW(reference.parameters(), lr=1e-5, betas=(.9, .95),
                                  weight_decay=.1, foreach=True, capturable=False, fused=False)
        native.load_state_dict(saved)
        self.assertFalse(native.param_groups[0]['capturable'])
        self.assertTrue(native.param_groups[0]['foreach'])
        self.assertFalse(native.param_groups[0]['fused'])
        for state in native.state.values():
            self.assertEqual(state['step'].device.type, 'cpu')
            self.assertEqual(float(state['step']), 7)
            self.assertEqual(state['exp_avg'].dtype, torch.float32)
        optimizer.zero_grad(set_to_none=True)
        native.zero_grad(set_to_none=True)
        optimizer.param_groups[0]['lr'] = before_lr
        x = torch.randn(2, 4)
        model(x).square().mean().backward()
        reference(x).square().mean().backward()
        optimizer.step()
        native.step()
        for a, b in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_safe_boundary_applies_exactly_one_good_update(self):
        model, optimizer = self.toy_state()
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4)).square().mean()
        loss.backward()
        calls = []
        def apply():
            calls.append(1)
            optimizer.step()
        norm = comparison.checked_step(model, optimizer, loss.detach(), apply, torch, 8)
        self.assertEqual(len(calls), 1)
        self.assertTrue(bool(torch.isfinite(norm)))
        for state in optimizer.state.values():
            self.assertEqual(float(state['step']), 8)

    def test_bad_gradient_keeps_parameter_and_adam_state_unchanged(self):
        model, optimizer = self.toy_state()
        weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        before = comparison.state_audit(optimizer.state_dict(), torch, 7)
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4)).square().mean()
        loss.backward()
        next(model.parameters()).grad.reshape(-1)[0] = float('nan')
        calls = []
        with self.assertRaises(comparison.DiagnosticNonfinite) as caught:
            comparison.checked_step(model, optimizer, loss.detach(), lambda: calls.append(1), torch, 8)
        self.assertEqual(calls, [])
        self.assertFalse(caught.exception.details['optimizer_applied'])
        self.assertGreater(len(caught.exception.details['nonfinite_gradients']), 0)
        self.assertEqual(comparison.state_audit(optimizer.state_dict(), torch, 7)['combined_tensor_sha256'],
                         before['combined_tensor_sha256'])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))

    def test_bad_loss_keeps_optimizer_untouched(self):
        model, optimizer = self.toy_state()
        calls = []
        with self.assertRaises(comparison.DiagnosticNonfinite):
            comparison.checked_step(model, optimizer, torch.tensor(float('inf')), lambda: calls.append(1), torch, 8)
        self.assertEqual(calls, [])
        self.assertEqual(comparison.state_audit(optimizer.state_dict(), torch, 7)['step_values'], [7.0])

    def test_vectorized_and_standard_data_paths_exact_on_full_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(53, dtype=np.uint32)
            np.save(root / 'train_tokens.npy', values)
            (root / 'manifest.json').write_text(json.dumps({'status': 'ready', 'block_length': 8,
                                                           'splits': {'train': {'tokens': 53}}}))
            data = PackedTokens(root, include_tail=True, pad_token_id=99)
            data.explicit_mask = True
            provider = PackedTokenBatches(device='cpu')
            for indices in ([1, 4, 0], [6, 2, 5], [6]):
                actual, targets, tokens = provider(data, indices)
                standard = {key: torch.stack([data[index][key] for index in indices])
                            for key in ('input_ids', 'labels')}
                for key in standard:
                    self.assertTrue(torch.equal(actual[key], standard[key]))
                self.assertEqual(targets, int((standard['labels'][:, 1:] != -100).sum()))
                self.assertEqual(tokens, int((standard['labels'] != -100).sum()))
                self.assertTrue(torch.equal(actual['attention_mask'], torch.ones_like(standard['input_ids'])))
                # Full blocks alias only in the fast representation; token values remain equal.
                if 6 not in indices:
                    self.assertEqual(actual['input_ids'].data_ptr(), actual['labels'].data_ptr())
                    self.assertNotEqual(standard['input_ids'].data_ptr(), standard['labels'].data_ptr())

    def test_fixed_probe_sampling_does_not_perturb_training_rng(self):
        torch.manual_seed(71)
        state = torch.get_rng_state()
        first = np.sort(np.random.default_rng(20261009).choice(3000, 256, replace=False))
        second = np.sort(np.random.default_rng(20261009).choice(3000, 256, replace=False))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_order_prefix_counts_global_cursor_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            orders = EpochOrders.create(Path(directory) / 'orders', blocks=7, batch=3, epochs=2, seed=42)
            for completed in range(7):
                indices = [int(index) for step in range(completed) for index in orders.indices(step)]
                inputs = sum(8 if index != 6 else 5 for index in indices)
                targets = sum(7 if index != 6 else 4 for index in indices)
                self.assertEqual(prefix_counts(orders, completed, 3, 8, 53), (inputs, targets, completed))


if __name__ == '__main__':
    unittest.main()
