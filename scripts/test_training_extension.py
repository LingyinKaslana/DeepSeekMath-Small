"""CPU proof tests for append-only completed-pilot CPT extensions."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from cpt_orders import EpochOrders, EXTENSION_GENERATOR, prefix_counts
from packed_data import PackedTokens
from source_migration import validate_lineage, verify_lineage_snapshots
from standard_native import prepare_train_probe
from train_cpt import learning_rate, validate_resume
from training_extension import (KIND, apply_training_extension, authorize_extension_signatures,
                                constant_schedule, sha, verify_data_prefix, verify_extension_snapshots)
from verify_cpt_checkpoint import verify_signature_and_counters


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old_data, self.new_data = self.root / "old_data", self.root / "new_data"
        self.old_run, self.output = self.root / "old_run", self.root / "extension"
        self.checkpoint = self.old_run / "final_checkpoint"
        for path in (self.old_data, self.new_data, self.checkpoint, self.old_run / "source", self.output / "source"):
            path.mkdir(parents=True)
        self.backend = json.loads((ROOT / "configs/cpt_backend_native.json").read_text())
        self.backend.update(sequence_length=4, effective_batch=4, micro_batch=4)
        self.backend["options"]["compile_shape"] = [4, 4]
        write(self.old_run / "backend_config.json", self.backend)
        shutil.copy2(self.old_run / "backend_config.json", self.output / "backend_config.json")
        self.old_manifest = {"status": "ready", "tokenizer": self.backend["model"],
                             "tokenizer_revision": self.backend["revision"], "block_length": 4,
                             "splits": {"train": {"tokens": 96}, "validation": {"tokens": 12}}}
        self.new_manifest = copy.deepcopy(self.old_manifest)
        self.new_manifest["splits"]["train"]["tokens"] = 237
        for directory, manifest, tokens in ((self.old_data, self.old_manifest, 96),
                                             (self.new_data, self.new_manifest, 237)):
            np.save(directory / "train_tokens.npy", np.arange(tokens, dtype=np.uint32))
            np.save(directory / "validation_tokens.npy", np.arange(12, dtype=np.uint32))
            write(directory / "manifest.json", manifest)
        shutil.copy2(self.old_data / "manifest.json", self.old_run / "data_manifest.json")
        shutil.copy2(self.new_data / "manifest.json", self.output / "data_manifest.json")
        (self.old_run / "source" / "toy.py").write_text("# original source\n")
        (self.output / "source" / "toy.py").write_text("# explicit extension implementation\n")
        self.before = self.signature(self.old_data, self.old_run, 1)
        self.after = self.signature(self.new_data, self.output, 2)
        self.old_orders = EpochOrders.create(self.checkpoint / "orders", 24, 4, 1, 42)
        old_dataset = PackedTokens(self.old_data, sequence_length=4, include_tail=True, pad_token_id=0)
        probe = prepare_train_probe(old_dataset, 3, 77)
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=3e-6, betas=(.9, .95), weight_decay=.1,
                                      eps=1e-8, foreach=True, capturable=False, fused=False)
        for _ in range(5):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        self.resumed = {"schema_version": 1, "signature": self.before,
                        "schedule": constant_schedule(5, self.before), "completed_steps": 5,
                        "trained_tokens": 80, "supervised_tokens": 60, "optimizer_updates": 5,
                        "order_metadata": self.old_orders.metadata,
                        "epoch_cursor": {"completed_epochs": 0, "step_in_epoch": 5},
                        "original_signature": self.before, "source_migrations": [],
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.get_rng_state(),
                        "optimizer": optimizer.state_dict(), "train_probe": probe,
                        "train_probe_history": [{"step": 0, "loss": 2.5}],
                        "validation_before": {"loss": 2.4}, "validation_history": [{"step": 5, "loss": 2.0}]}
        torch.save(self.resumed, self.checkpoint / "training_state.pt")
        source_manifest = {key: value for key, value in self.resumed.items()
                           if key not in ("optimizer", "cpu_rng", "cuda_rng")}
        write(self.checkpoint / "checkpoint_manifest.json", source_manifest)
        (self.checkpoint / "model.safetensors").write_bytes(b"immutable toy serialized weights")
        self.policy = {"schema_version": 1, "kind": KIND, "source_checkpoint": str(self.checkpoint),
                       "source_data": str(self.old_data), "target_data": str(self.new_data),
                       "source_checkpoint_manifest_sha256": sha(self.checkpoint / "checkpoint_manifest.json"),
                       "source_training_state_sha256": sha(self.checkpoint / "training_state.pt"),
                       "source_weight_sha256": {"model.safetensors": sha(self.checkpoint / "model.safetensors")},
                       "completed_steps": 5, "trained_tokens": 80, "supervised_tokens": 60,
                       "optimizer_updates": 5, "target_steps": 30, "target_epochs": 2,
                       "target_input_tokens": 474, "reason": "Explicit authorized CPU extension test"}
        self.policy_path = self.root / "policy.json"
        write(self.policy_path, self.policy)
        self.args = argparse.Namespace(training_extension_file=self.policy_path, resume=self.checkpoint,
                                       data=self.new_data, output=self.output, backend_config=self.output / "backend_config.json",
                                       epochs=2, steps=30, lr=3e-6, seed=42, warmup_steps=2, lr_schedule="constant",
                                       allow_source_migration=False, allow_native_backend_migration=False)
        self.dataset = PackedTokens(self.new_data, sequence_length=4, include_tail=True, pad_token_id=0)

    def tearDown(self):
        self.temporary.cleanup()

    def signature(self, data, run, epochs):
        return {"backend_config_sha256": sha(run / "backend_config.json"),
                "data_manifest_sha256": sha(data / "manifest.json"),
                "token_file_sha256": {name: sha(data / name) for name in ("train_tokens.npy", "validation_tokens.npy")},
                "peak_lr": 3e-6, "seed": 42, "sequence_length": 4, "effective_batch": 4,
                "epochs": epochs, "warmup_steps": 2, "lr_schedule": "constant",
                "source_sha256": {"toy.py": sha(run / "source" / "toy.py")}}

    def apply(self):
        state = {key: value for key, value in self.resumed.items()
                 if key not in ("schema_version", "optimizer", "cpu_rng", "cuda_rng")}
        orders = apply_training_extension(state, self.resumed, self.after, self.args, self.old_orders, self.dataset)
        return state, orders

    def test_append_extension_preserves_history_and_all_source_bytes(self):
        original_files = {str(path): sha(path) for path in self.old_run.rglob("*") if path.is_file()}
        cpu_rng = torch.get_rng_state().clone()
        state, orders = self.apply()
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(original_files, {str(path): sha(path) for path in self.old_run.rglob("*") if path.is_file()})
        self.assertEqual(state["completed_steps"], 5)
        self.assertEqual(state["trained_tokens"], 80)
        self.assertEqual(state["supervised_tokens"], 60)
        self.assertIs(state["validation_history"], self.resumed["validation_history"])
        self.assertEqual(state["validation_before"], self.resumed["validation_before"])
        self.assertEqual(state["train_probe"], self.resumed["train_probe"])
        self.assertTrue(np.array_equal(orders.orders[0][:20], self.old_orders.orders[0][:20]))
        self.assertEqual(prefix_counts(orders, 30, 4, 4, 237), (474, 354, 30))
        for step in range(5):
            self.assertEqual(learning_rate(state["schedule"], step), learning_rate(self.resumed["schedule"], step))
        self.assertEqual(learning_rate(state["schedule"], 5), 3e-6)
        self.assertEqual(verify_extension_snapshots(state["training_extension"])["status"], "passed")
        self.assertEqual(verify_lineage_snapshots(state, sha)["training_extension_count"], 1)
        probe = prepare_train_probe(self.dataset, 3, 77, state["train_probe"], preserve_existing_indices=True)
        self.assertEqual(probe, self.resumed["train_probe"])
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_train_probe(self.dataset, 3, 77, state["train_probe"])

    def test_normal_resume_after_extension_and_independent_counter_verifier(self):
        state, orders = self.apply()
        args = copy.copy(self.args)
        args.training_extension_file = None
        validate_resume(dict(state, schema_version=1), self.after, self.dataset, args, orders)
        # A subsequent checkpoint can resume normally, including across the
        # new epoch boundary, without applying the extension a second time.
        for step in (6, 15, 16, 29):
            resumed = copy.deepcopy(state)
            resumed["completed_steps"] = step
            resumed["trained_tokens"], resumed["supervised_tokens"], resumed["optimizer_updates"] = prefix_counts(orders, step, 4, 4, 237)
            resumed["epoch_cursor"] = {"completed_epochs": step // 15, "step_in_epoch": step % 15}
            validate_resume(dict(resumed, schema_version=1), self.after, self.dataset, args, orders)
        saved = dict(state, schema_version=1, cpu_rng=self.resumed["cpu_rng"], cuda_rng=self.resumed["cuda_rng"])
        new_checkpoint = self.output / "stopped_checkpoint"
        new_checkpoint.mkdir()
        orders.copy_to(new_checkpoint / "orders")
        checkpoint_manifest = copy.deepcopy(state)
        checkpoint_manifest.update(schema_version=1, native_retry_count=0)
        write(new_checkpoint / "checkpoint_manifest.json", checkpoint_manifest)
        write(self.output / "manifest.json", {"signature": self.after, "original_signature": self.before,
                                               "source_migrations": state["source_migrations"],
                                               "training_extension": state["training_extension"],
                                               "extension_source_checkpoint": str(self.checkpoint),
                                               "arguments": {"lr_schedule": "constant", "warmup_steps": 2}})
        verified = verify_signature_and_counters(new_checkpoint, saved, checkpoint_manifest)
        self.assertEqual(verified["trained_tokens"], 80)
        self.assertEqual(verified["training_extension_count"], 1)

    def test_reject_protocol_changes_and_validation_change(self):
        changes = {"backend_config_sha256": "0" * 64, "peak_lr": 1e-5, "seed": 43,
                   "sequence_length": 8, "effective_batch": 2, "warmup_steps": 3,
                   "lr_schedule": "multistep", "epochs": 3}
        for key, value in changes.items():
            bad = copy.deepcopy(self.after)
            bad[key] = value
            with self.assertRaises(ValueError, msg=key):
                authorize_extension_signatures(self.before, bad)
        bad = copy.deepcopy(self.after)
        bad["token_file_sha256"]["validation_tokens.npy"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "validation"):
            authorize_extension_signatures(self.before, bad)

    def test_reject_changed_original_token_prefix_and_partial_old_block(self):
        values = np.arange(237, dtype=np.uint32)
        values[12] += 1
        np.save(self.new_data / "train_tokens.npy", values)
        changed = self.signature(self.new_data, self.output, 2)
        with self.assertRaisesRegex(ValueError, "prefix"):
            verify_data_prefix(self.old_data, self.new_data, self.before, changed)
        values[12] -= 1
        np.save(self.new_data / "train_tokens.npy", values)
        np.save(self.old_data / "train_tokens.npy", np.arange(95, dtype=np.uint32))
        self.old_manifest["splits"]["train"]["tokens"] = 95
        write(self.old_data / "manifest.json", self.old_manifest)
        changed_before = self.signature(self.old_data, self.old_run, 1)
        with self.assertRaisesRegex(ValueError, "partial block"):
            verify_data_prefix(self.old_data, self.new_data, changed_before, self.after)

    def test_reject_source_weight_change_and_wrong_budget(self):
        (self.checkpoint / "model.safetensors").write_bytes(b"changed serialized weights")
        with self.assertRaisesRegex(ValueError, "weights"):
            self.apply()
        (self.checkpoint / "model.safetensors").write_bytes(b"immutable toy serialized weights")
        self.args.steps = 29
        with self.assertRaisesRegex(ValueError, "budget"):
            self.apply()

    def test_reject_forged_prefix_or_biased_future_permutation(self):
        state, orders = self.apply()
        path = orders.directory / orders.metadata["files"][0]["name"]
        values = np.load(path).copy()
        values[20], values[21] = values[21], values[20]
        np.save(path, values)
        metadata = copy.deepcopy(orders.metadata)
        metadata["files"][0]["sha256"] = sha(path)
        metadata["files"][0]["tail_position"] = int(np.flatnonzero(values == orders.blocks - 1)[0])
        with self.assertRaisesRegex(ValueError, "uniform complement"):
            EpochOrders(orders.directory, metadata)
        original = np.load(self.checkpoint / "orders" / "order_epoch_00000.npy").copy()
        forged = copy.deepcopy(state)
        forged["source_migrations"][0]["to_signature"]["peak_lr"] = 1e-5
        with self.assertRaises(ValueError):
            validate_lineage(forged)

    def test_production_budget_prefix_and_uniform_future(self):
        previous = EpochOrders.create(self.root / "actual_old_orders", 250000, 16, 1, 42)
        orders = EpochOrders.extend_completed_prefix(self.root / "actual_new_orders", previous, 15259,
                                                     2441407, 16, 2, 42)
        self.assertEqual(orders.metadata["generator"], EXTENSION_GENERATOR)
        self.assertTrue(np.array_equal(orders.orders[0][:244144], previous.orders[0][:244144]))
        self.assertEqual(prefix_counts(orders, 15259, 16, 2048, 5_000_000_000),
                         (500_006_912, 499_762_768, 15_259))
        self.assertEqual(prefix_counts(orders, 305176, 16, 2048, 5_000_000_000),
                         (10_000_000_000, 9_995_117_186, 305_176))
        self.assertTrue(np.any(orders.orders[0][244144:244144 + 16] >= 250000))
        EpochOrders(orders.directory, orders.metadata)


if __name__ == "__main__":
    unittest.main()
