"""CPU-only regression checks for native retry and audited source migration."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cpt_orders import EpochOrders, prefix_counts
from nonfinite_retry import NonfiniteUpdateError, checked_optimizer_update
from source_migration import (apply_source_migration, authorize_signature, validate_lineage,
                              verify_lineage_snapshots)
from train_cpt import make_schedule, parser, update, validate_resume


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.01, betas=(.9, .95), weight_decay=.1)
        self.updates = 0

    def step(self, closure=None):
        self.updates += 1
        return super().step(closure)


def backward(model, values, mode="finite", draws=None):
    if draws is not None:
        draws.append(torch.rand(5))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(values).float().square().mean()
    loss.backward()
    if mode == "bad_norm":
        next(model.parameters()).grad.reshape(-1)[0] = float("nan")
    return loss.detach() if mode != "bad_loss" else torch.tensor(float("nan"))


class RetryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
        self.values = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
        self.optimizer = CountingAdamW(self.model.parameters())

    def assert_parameters_equal(self, snapshots):
        for parameter, expected in zip(self.model.parameters(), snapshots):
            self.assertTrue(torch.equal(parameter.detach(), expected))

    def test_disabled_retry_keeps_strict_failure_and_zero_updates(self):
        snapshots = [parameter.detach().clone() for parameter in self.model.parameters()]
        with self.assertRaises(NonfiniteUpdateError):
            checked_optimizer_update(self.model, self.optimizer,
                                     lambda: backward(self.model, self.values, "bad_norm"))
        self.assertEqual(self.optimizer.updates, 0)
        self.assertEqual(len(self.optimizer.state), 0)
        self.assert_parameters_equal(snapshots)

    def test_one_native_retry_same_rng_exact_single_update(self):
        reference = copy.deepcopy(self.model)
        reference_optimizer = CountingAdamW(reference.parameters())
        initial_rng = torch.get_rng_state().clone()
        expected_loss, expected_norm = checked_optimizer_update(
            reference, reference_optimizer, lambda: backward(reference, self.values))
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        draws, events, optimizer_calls = [], [], []

        def single_step():
            optimizer_calls.append(1)
            self.optimizer.step()

        # Draws belong to both attempts, so also include them in the reference.
        torch.set_rng_state(initial_rng)
        reference = copy.deepcopy(self.model)
        reference_optimizer = CountingAdamW(reference.parameters())
        expected_loss, expected_norm = checked_optimizer_update(
            reference, reference_optimizer, lambda: backward(reference, self.values, draws=[]))
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        loss, norm = checked_optimizer_update(
            self.model, self.optimizer,
            lambda: backward(self.model, self.values, "bad_norm", draws),
            native_backward=lambda: backward(self.model, self.values, draws=draws),
            retry_native=True, step_callable=single_step, step=17, event_callback=events.append)
        self.assertTrue(torch.equal(draws[0], draws[1]))
        self.assertTrue(torch.equal(loss, expected_loss))
        self.assertTrue(torch.equal(norm, expected_norm))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for actual, expected in zip(self.model.parameters(), reference.parameters()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(self.optimizer.updates, 1)
        self.assertEqual(len(optimizer_calls), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 17)
        self.assertEqual(events[0]["cause"], "gradient_norm")
        self.assertEqual(events[0]["fast_grad_norm"], "nan")
        self.assertEqual(events[0]["fast_nonfinite_gradients"]["nonfinite_tensor_count"], 1)
        self.assertEqual(events[0]["fast_nonfinite_gradients"]["nonfinite_element_count"], 1)
        self.assertEqual(events[0]["fast_nonfinite_gradients"]["tensors"][0]["name"], "0.weight")
        self.assertTrue(math.isfinite(events[0]["native_grad_norm"]))

    def test_nonfinite_loss_with_finite_gradient_also_retries(self):
        events = []
        loss, norm = checked_optimizer_update(
            self.model, self.optimizer, lambda: backward(self.model, self.values, "bad_loss"),
            native_backward=lambda: backward(self.model, self.values),
            retry_native=True, event_callback=events.append)
        self.assertEqual(self.optimizer.updates, 1)
        self.assertEqual(events[0]["cause"], "loss")
        self.assertEqual(events[0]["fast_loss"], "nan")
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(norm)))

    def test_native_failure_keeps_weights_moments_rng_and_zero_updates(self):
        # Initialize real Adam moments, then prove the failed second update
        # leaves these existing tensors and the parameter values unchanged.
        checked_optimizer_update(self.model, self.optimizer, lambda: backward(self.model, self.values))
        self.optimizer.updates = 0
        snapshots = [parameter.detach().clone() for parameter in self.model.parameters()]
        state_before = copy.deepcopy(self.optimizer.state_dict())
        rng_before = torch.get_rng_state().clone()
        events = []
        with self.assertRaises(NonfiniteUpdateError):
            checked_optimizer_update(
                self.model, self.optimizer, lambda: backward(self.model, self.values, "bad_norm"),
                native_backward=lambda: backward(self.model, self.values, "bad_loss"),
                retry_native=True, event_callback=events.append)
        self.assertEqual(self.optimizer.updates, 0)
        self.assert_parameters_equal(snapshots)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        for index, values in state_before["state"].items():
            for name, expected in values.items():
                self.assertTrue(torch.equal(self.optimizer.state_dict()["state"][index][name], expected))
        self.assertEqual(events[0]["status"], "failed_before_optimizer")
        self.assertFalse(events[0]["optimizer_applied"])
        self.assertEqual(events[0]["native_nonfinite_gradients"]["nonfinite_tensor_count"], 0)

    def test_unrelated_runtime_error_does_not_retry(self):
        native_calls = []

        def broken():
            raise RuntimeError("Unrelated kernel failure")

        with self.assertRaisesRegex(RuntimeError, "Unrelated"):
            checked_optimizer_update(self.model, self.optimizer, broken,
                                     native_backward=lambda: native_calls.append(1), retry_native=True)
        self.assertEqual(native_calls, [])
        self.assertEqual(self.optimizer.updates, 0)

    def test_trainer_reuses_batches_and_keeps_counts(self):
        # Redirect only the trainer's CUDA tensor/autocast construction to CPU.
        # This exercises its actual batching/retry integration without CUDA.
        provider_indices, attempted_values, native_context_entries = [], [], []
        real_zeros, real_autocast = torch.zeros, torch.autocast
        model = self.model

        def provider(dataset, indices):
            provider_indices.append(tuple(indices))
            tensor = self.values[:len(indices)]
            return {"input_ids": tensor, "labels": tensor}, len(indices) * 3, len(indices) * 4

        def forward(*, input_ids, labels, **kwargs):
            attempted_values.append(id(input_ids))
            loss = model(input_ids).float().square().mean()
            # Finite scalar with a nonfinite weight gradient, as in the GPU
            # production check; the failed attempt never reaches the optimizer.
            next(model.parameters()).register_hook(inject_once)
            return SimpleNamespace(loss=loss)

        calls = [0]

        def inject_once(gradient):
            calls[0] += 1
            if calls[0] == 1:
                gradient = gradient.clone()
                gradient.reshape(-1)[0] = float("nan")
            return gradient

        @contextmanager
        def native_context(_):
            native_context_entries.append(1)
            original_forward = model.forward

            def native(*, input_ids, labels, **kwargs):
                attempted_values.append(id(input_ids))
                return SimpleNamespace(loss=original_forward(input_ids).float().square().mean())

            model.forward = native
            try:
                yield model
            finally:
                model.forward = original_forward

        def cpu_zeros(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return real_zeros(*args, **kwargs)

        with patch("train_cpt.torch.zeros", cpu_zeros), patch("train_cpt.torch.autocast", lambda *a, **k: real_autocast("cpu", dtype=torch.bfloat16)), patch("train_cpt.eager_native", native_context):
            loss, norm, inputs, targets, applied = update(
                model, forward, self.optimizer, object(), provider, [5, 2, 9], 3, "causal",
                retry_native_on_nonfinite=True, step=1)
        self.assertEqual(provider_indices, [(5, 2, 9)])
        self.assertEqual(attempted_values[0], attempted_values[1])
        self.assertEqual(native_context_entries, [1])
        self.assertEqual((inputs, targets, applied), (12, 9, True))
        self.assertEqual(self.optimizer.updates, 1)
        self.assertTrue(bool(torch.isfinite(loss) & torch.isfinite(norm)))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original = {"source_sha256": {"train_cpt.py": "a" * 64},
                         "backend_config_sha256": "c" * 64, "token_file_sha256": {"train": "d" * 64},
                         "peak_lr": 1e-5, "seed": 42, "sequence_length": 4, "effective_batch": 3, "epochs": 2}
        self.current = copy.deepcopy(self.original)
        self.current["source_sha256"] = {"train_cpt.py": "b" * 64, "nonfinite_retry.py": "e" * 64}

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_flags_and_source_only_authorization(self):
        args = parser().parse_args(["--backend-config", "b", "--data", "d", "--output", "o"])
        self.assertFalse(args.retry_native_on_nonfinite)
        self.assertFalse(args.allow_source_migration)
        with self.assertRaises(ValueError):
            authorize_signature(self.original, self.current)
        self.assertTrue(authorize_signature(self.original, self.current, True))
        self.assertFalse(authorize_signature(self.current, self.current, True))
        for key in ("backend_config_sha256", "token_file_sha256", "peak_lr", "seed", "sequence_length", "effective_batch", "epochs"):
            bad = copy.deepcopy(self.current)
            bad[key] = "changed"
            with self.assertRaises(ValueError, msg=key):
                authorize_signature(self.original, bad, True)
        bad = dict(self.current, new_policy_field=True)
        with self.assertRaises(ValueError):
            authorize_signature(self.original, bad, True)

    def test_continuous_original_signature_audit_and_tampering_rejection(self):
        state = {"signature": self.original, "completed_steps": 17}
        apply_source_migration(state, self.current, self.root / "old" / "checkpoint", self.root / "new")
        original, migrations = validate_lineage(state)
        self.assertEqual(original, self.original)
        self.assertEqual(len(migrations), 1)
        self.assertEqual(migrations[0]["changed_fields"], ["source_sha256"])
        bad = copy.deepcopy(state)
        bad["source_migrations"][0]["to_signature"]["seed"] = 43
        with self.assertRaises(ValueError):
            validate_lineage(bad)
        bad = copy.deepcopy(state)
        bad["source_migrations"][0]["completed_steps"] = 18
        with self.assertRaises(ValueError):
            validate_lineage(bad)

    def test_migration_snapshot_hashes_are_checked(self):
        old_source, new_source = self.root / "old" / "source", self.root / "new" / "source"
        old_source.mkdir(parents=True)
        new_source.mkdir(parents=True)
        for source, value in ((old_source, "old"), (new_source, "new")):
            (source / "train_cpt.py").write_text(value)

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        original, current = copy.deepcopy(self.original), copy.deepcopy(self.current)
        original["source_sha256"] = {"train_cpt.py": sha(old_source / "train_cpt.py")}
        current["source_sha256"] = {"train_cpt.py": sha(new_source / "train_cpt.py")}
        state = {"signature": original, "completed_steps": 3}
        apply_source_migration(state, current, self.root / "old" / "checkpoint", self.root / "new")
        self.assertEqual(verify_lineage_snapshots(state, sha)["source_only_migration_count"], 1)
        (old_source / "train_cpt.py").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
            verify_lineage_snapshots(state, sha)

    def test_resume_keeps_counters_order_and_lr_strict(self):
        orders = EpochOrders.create(self.root / "orders", 11, 3, 2, 42)
        dataset = SimpleNamespace(tokens=range(41), sequence_length=4)
        dataset = type("Dataset", (), {"tokens": range(41), "sequence_length": 4, "__len__": lambda _: 11})()
        inputs, targets, updates = prefix_counts(orders, 3, 3, 4, 41)
        state = {"schema_version": 1, "signature": self.original, "completed_steps": 3,
                 "schedule": make_schedule(8, 1e-5), "trained_tokens": inputs,
                 "supervised_tokens": targets, "optimizer_updates": updates}
        args = argparse.Namespace(epochs=2, steps=None, lr=1e-5, seed=42, allow_source_migration=True)
        validate_resume(state, self.current, dataset, args, orders)
        for key in ("trained_tokens", "supervised_tokens", "optimizer_updates"):
            bad = dict(state, **{key: state[key] + 1})
            with self.assertRaisesRegex(ValueError, "counters"):
                validate_resume(bad, self.current, dataset, args, orders)
        bad = copy.deepcopy(state)
        bad["schedule"]["peak_lr"] = 2e-5
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_resume(bad, self.current, dataset, args, orders)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
