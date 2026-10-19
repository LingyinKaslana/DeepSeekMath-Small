"""CPU checks for compact orders, multi-epoch tails, scheduling and resume gates."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from cpt_orders import EpochOrders, load_saved_orders, order_indices, prefix_counts
from train_cpt import epoch_cursor, has_full_compile_batch, learning_rate, make_schedule, prune_periodic_checkpoints, validate_resume
from verify_cpt_pipeline import verify as verify_pipeline


class FakeDataset:
    def __init__(self, tokens, length):
        self.tokens = range(tokens)
        self.sequence_length = length

    def __len__(self):
        return math.ceil(len(self.tokens) / self.sequence_length)


def direct_prefix(orders, steps, batch, length, tokens):
    inputs = targets = updates = 0
    for step in range(steps):
        indices = order_indices(orders, step, batch)
        real_lengths = [min(length, tokens - int(index) * length) for index in indices]
        inputs += sum(real_lengths)
        step_targets = sum(max(n - 1, 0) for n in real_lengths)
        targets += step_targets
        updates += int(step_targets > 0)
    return inputs, targets, updates


class EpochTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def orders(self, name, tokens, length, batch, epochs=2, seed=42):
        return EpochOrders.create(self.root / name, math.ceil(tokens / length), batch, epochs, seed)

    def test_historical_first_shuffle_and_distinct_second(self):
        orders = self.orders("original", 100, 4, 3)
        expected = torch.randperm(25, generator=torch.Generator().manual_seed(42)).numpy()
        self.assertTrue(np.array_equal(orders.orders[0], expected))
        self.assertFalse(np.array_equal(orders.orders[0], orders.orders[1]))
        self.assertIsInstance(orders.orders[0], np.memmap)
        self.assertEqual(orders.orders[0].dtype, np.int32)
        orders.copy_to(self.root / "checkpoint" / "orders")
        restored = load_saved_orders({"order_metadata": orders.metadata}, self.root / "checkpoint")
        self.assertEqual(restored.metadata, orders.metadata)
        for a, b in zip(orders.orders, restored.orders):
            self.assertTrue(np.array_equal(a, b))

    def test_all_tail_prefixes_and_zero_target_updates(self):
        cases = 0
        for length in (2, 4, 7):
            for tokens in (1, length, length + 1, length * 4 - 1, length * 4 + 1, length * 5 + 2):
                for batch in (1, 2, 4, 16):
                    orders = self.orders(f"case_{cases}", tokens, length, batch, epochs=3, seed=cases)
                    for step in range(orders.epoch_steps * 3 + 1):
                        expected = direct_prefix(orders, step, batch, length, tokens)
                        self.assertEqual(prefix_counts(orders, step, batch, length, tokens), expected,
                                         (tokens, length, batch, step))
                    for epoch in range(3):
                        complete = prefix_counts(orders, (epoch + 1) * orders.epoch_steps, batch, length, tokens)
                        self.assertEqual(complete[:2], ((epoch + 1) * tokens,
                                                      (epoch + 1) * (tokens - orders.blocks)))
                    cases += 1
        self.assertEqual(cases, 72)

    def test_legacy_list_counter_and_rejection(self):
        for tokens in (1, 17, 18, 20):
            blocks = math.ceil(tokens / 4)
            order = torch.randperm(blocks, generator=torch.Generator().manual_seed(1)).tolist()
            for batch in (1, 2, 4):
                for steps in range(math.ceil(blocks / batch) + 1):
                    self.assertEqual(prefix_counts(order, steps, batch, 4, tokens),
                                     direct_prefix(order, steps, batch, 4, tokens))
            self.assertEqual(load_saved_orders({"order": order}, self.root), order)
        for bad in ([0, 0], [True], [1]):
            with self.assertRaises(ValueError):
                load_saved_orders({"order": bad}, self.root)

    def test_checksum_and_tail_metadata_rejection(self):
        orders = self.orders("checks", 37, 4, 3)
        bad_metadata = copy.deepcopy(orders.metadata)
        bad_metadata["files"][0]["tail_position"] = (bad_metadata["files"][0]["tail_position"] + 1) % orders.blocks
        with self.assertRaisesRegex(ValueError, "tail position"):
            EpochOrders(orders.directory, bad_metadata)
        path = orders.directory / orders.metadata["files"][0]["name"]
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "checksum"):
            EpochOrders(orders.directory, orders.metadata)

    def test_masked_tail_does_not_claim_compile_initialization(self):
        dataset = FakeDataset(17, 4)
        self.assertFalse(has_full_compile_batch([0, 1, 2, 4], dataset, 4, "same_as_input"))
        self.assertTrue(has_full_compile_batch([0, 1, 2, 3], dataset, 4, "same_as_input"))
        self.assertTrue(has_full_compile_batch([0, 4, 1, 2], dataset, 2, "same_as_input"))
        self.assertTrue(has_full_compile_batch([0, 1, 2, 4], dataset, 4, "any"))
        self.assertFalse(has_full_compile_batch([0, 1], dataset, 4, "any"))

    def test_resume_on_both_sides_of_epoch_boundary_and_global_lr(self):
        dataset = FakeDataset(41, 4)
        orders = self.orders("resume", 41, 4, 3)
        signature = {"effective_batch": 3, "epochs": 2}
        args = argparse.Namespace(epochs=2, steps=None, lr=1e-5, seed=42)
        schedule = make_schedule(orders.epoch_steps * 2, args.lr)
        rates = [learning_rate(schedule, step) for step in range(schedule["total_steps"])]
        for step in (0, orders.epoch_steps - 1, orders.epoch_steps, orders.epoch_steps + 1, schedule["total_steps"] - 1):
            inputs, targets, updates = prefix_counts(orders, step, 3, 4, 41)
            state = {"schema_version": 1, "signature": signature, "schedule": schedule,
                     "completed_steps": step, "trained_tokens": inputs, "supervised_tokens": targets,
                     "optimizer_updates": updates, "epoch_cursor": epoch_cursor(step, orders.epoch_steps)}
            validate_resume(state, signature, dataset, args, orders)
            self.assertEqual(rates[step:], [learning_rate(state["schedule"], k)
                                          for k in range(step, schedule["total_steps"])])
            bad = dict(state, trained_tokens=inputs + 1)
            with self.assertRaisesRegex(ValueError, "counters"):
                validate_resume(bad, signature, dataset, args, orders)
            bad = dict(state, epoch_cursor={"completed_epochs": 9, "step_in_epoch": 0})
            with self.assertRaisesRegex(ValueError, "cursor"):
                validate_resume(bad, signature, dataset, args, orders)
        self.assertEqual(prefix_counts(orders, schedule["total_steps"], 3, 4, 41), (82, 60, 8))
        self.assertEqual(rates[orders.epoch_steps], args.lr)
        # Historical ceil-based decay boundaries can leave no 0.1 segment in
        # a tiny eight-update smoke; check that segment on a longer plan.
        self.assertEqual(learning_rate(make_schedule(20, args.lr), 19), args.lr * .1)
        args.steps = 3
        with self.assertRaisesRegex(ValueError, "schedule"):
            validate_resume(state, signature, dataset, args, orders)

    def test_five_billion_budget_without_large_python_lists(self):
        orders = self.orders("production", 5_000_000_000, 2048, 16)
        self.assertEqual(orders.blocks, 2_441_407)
        self.assertEqual(orders.epoch_steps, 152_588)
        self.assertEqual(prefix_counts(orders, orders.epoch_steps * 2, 16, 2048, 5_000_000_000),
                         (10_000_000_000, 9_995_117_186, 305_176))
        self.assertLess(sum(order.nbytes for order in orders.orders), 20_000_000)
        self.assertLess(len(json.dumps(orders.metadata)), 2000)
        self.assertEqual(len(orders.indices(orders.epoch_steps - 1)), 15)
        self.assertEqual(len(orders.indices(orders.epoch_steps)), 16)

    def test_periodic_retention_preserves_foreign_and_epoch_checkpoints(self):
        signature = {"source": "current"}
        for step in (1, 2, 3, 4):
            path = self.root / f"checkpoint_step_{step:08d}"
            path.mkdir()
            (path / "training_state.pt").write_bytes(b"test")
            (path / "checkpoint_manifest.json").write_text(json.dumps({"signature": signature, "completed_steps": step}))
        for name in ("final_checkpoint", "stopped_checkpoint", "checkpoint_epoch_0001", "checkpoint_step_00000005.incomplete"):
            (self.root / name).mkdir()
        foreign = self.root / "checkpoint_step_00000006"
        foreign.mkdir()
        (foreign / "training_state.pt").write_bytes(b"test")
        (foreign / "checkpoint_manifest.json").write_text(json.dumps({"signature": {"source": "foreign"}, "completed_steps": 6}))
        external = self.root / "external_resume"
        external.mkdir()
        (self.root / "checkpoint_step_00000007").symlink_to(external, target_is_directory=True)
        removed = prune_periodic_checkpoints(self.root, 2, signature)
        self.assertEqual(len(removed), 2)
        self.assertFalse((self.root / "checkpoint_step_00000001").exists())
        self.assertTrue((self.root / "checkpoint_step_00000003").exists())
        self.assertTrue((self.root / "checkpoint_epoch_0001").exists())
        self.assertTrue(foreign.exists())
        self.assertTrue(external.exists())
        self.assertTrue((self.root / "checkpoint_step_00000007").is_symlink())

    def test_multi_epoch_stop_resume_pipeline_evidence(self):
        from cpt_orders import sha256
        config = self.root / "backend.json"
        config.write_text('{"validated":true}')
        signature = {"backend_config_sha256": sha256(config), "sequence_length": 4,
                     "effective_batch": 3, "epochs": 2}
        orders = self.orders("evidence_orders", 41, 4, 3)
        schedule = make_schedule(8, 1e-5)
        rows = []
        for step in range(1, 9):
            before = prefix_counts(orders, step - 1, 3, 4, 41)
            after = prefix_counts(orders, step, 3, 4, 41)
            rows.append({"step": step, "lr": learning_rate(schedule, step - 1), "loss": 1.8,
                         "grad_norm": .5, "seconds": .3, "input_tokens": after[0] - before[0],
                         "supervised_tokens": after[1] - before[1], "cumulative_tokens": after[0],
                         "optimizer_applied": after[2] > before[2], "first_full_shape_update": step in (1, 5)})
        first_checkpoint = self.root / "first" / "stopped_checkpoint"
        for label, stop, initial, status in (("first", 4, 0, "stopped"), ("resumed", 8, 4, "complete")):
            run = self.root / label
            run.mkdir()
            checkpoint = run / ("stopped_checkpoint" if status == "stopped" else "final_checkpoint")
            checkpoint.mkdir()
            orders.copy_to(checkpoint / "orders")
            inputs, targets, updates = prefix_counts(orders, stop, 3, 4, 41)
            state = {"signature": signature, "schedule": schedule, "order_metadata": orders.metadata,
                     "completed_steps": stop, "trained_tokens": inputs, "supervised_tokens": targets,
                     "optimizer_updates": updates}
            torch.save(state, checkpoint / "training_state.pt")
            validation = {"step": stop, "loss": 1.8, "loss_backend": "native eager"}
            result = dict(state, status=status, checkpoint=str(checkpoint), dataset_input_tokens=41,
                          configured_input_tokens=82, validation_after=validation,
                          all_dataset_tokens_consumed=status == "complete")
            manifest = {"signature": signature, "schedule": schedule, "resumed_completed_steps": initial,
                        "arguments": {"steps": None, "resume": str(first_checkpoint) if initial else None}}
            (run / "result.json").write_text(json.dumps(result))
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "validation_before.json").write_text(json.dumps(dict(validation, step=initial)))
            (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[initial:stop]))
            (run / "backend_config.json").write_bytes(config.read_bytes())
        args = argparse.Namespace(first_run=self.root / "first", resumed_run=self.root / "resumed", backend_config=config)
        result = verify_pipeline(args)
        self.assertEqual(result["status"], "passed", [item for item in result["checks"] if not item["passed"]])
        self.assertEqual(result["check_count"], 24)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
