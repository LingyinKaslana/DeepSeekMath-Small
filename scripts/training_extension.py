"""Explicit, audited extension of completed constant-LR native CPT pilots.

The old checkpoint stays complete and immutable. Only append-only training
data, the epoch budget and snapshotted implementation may change. All old
updates, token counters, weights, Adam tensors and training RNG are retained.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from cpt_orders import EpochOrders, EXTENSION_GENERATOR, indices_sha256, prefix_counts


KIND = "completed_constant_native_prefix_extension"
POLICY_KEYS = {"schema_version", "kind", "source_checkpoint", "source_data", "target_data",
               "source_checkpoint_manifest_sha256", "source_training_state_sha256",
               "source_weight_sha256",
               "completed_steps", "trained_tokens", "supervised_tokens", "optimizer_updates",
               "target_steps", "target_epochs", "target_input_tokens", "reason"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def token_digest(tokens):
    digest = hashlib.sha256()
    for start in range(0, len(tokens), 1_048_576):
        digest.update(memoryview(np.ascontiguousarray(tokens[start:start + 1_048_576])).cast("B"))
    return digest.hexdigest()


def authorize_extension_signatures(before, after):
    require(isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after),
            "Training extension cannot add/remove training protocol fields")
    allowed = {"data_manifest_sha256", "token_file_sha256", "epochs", "source_sha256"}
    require(all(before[key] == after[key] for key in before if key not in allowed),
            "Training extension cannot change model/backend/LR/warmup/seed/context/batch")
    require(before.get("epochs") == 1 and after.get("epochs") == 2,
            "Training extension supports one pilot epoch to two expanded epochs only")
    require(before.get("lr_schedule") == after.get("lr_schedule") == "constant"
            and isinstance(before.get("warmup_steps"), int) and not isinstance(before["warmup_steps"], bool),
            "Training extension requires an explicitly fixed constant-LR warmup")
    old_hashes, new_hashes = before.get("token_file_sha256"), after.get("token_file_sha256")
    require(isinstance(old_hashes, dict) and isinstance(new_hashes, dict)
            and set(old_hashes) == set(new_hashes) == {"train_tokens.npy", "validation_tokens.npy"}
            and old_hashes["validation_tokens.npy"] == new_hashes["validation_tokens.npy"]
            and old_hashes["train_tokens.npy"] != new_hashes["train_tokens.npy"]
            and before.get("data_manifest_sha256") != after.get("data_manifest_sha256"),
            "Training extension must expand training data and preserve validation bytes")
    require(isinstance(before.get("source_sha256"), dict) and isinstance(after.get("source_sha256"), dict),
            "Training extension requires source snapshots")
    return True


def validate_policy(policy):
    require(isinstance(policy, dict) and set(policy) == POLICY_KEYS
            and policy["schema_version"] == 1 and policy["kind"] == KIND,
            "Unknown or incomplete training-extension policy")
    for key in ("completed_steps", "trained_tokens", "supervised_tokens", "optimizer_updates",
                "target_steps", "target_epochs", "target_input_tokens"):
        require(isinstance(policy[key], int) and not isinstance(policy[key], bool) and policy[key] > 0,
                f"Invalid training-extension counter: {key}")
    for key in ("source_checkpoint", "source_data", "target_data"):
        require(isinstance(policy[key], str) and Path(policy[key]).is_absolute(),
                f"Training-extension {key} must be an absolute path")
    require(isinstance(policy["reason"], str) and bool(policy["reason"].strip()),
            "Training extension must record its authorization reason")
    hashes = policy["source_weight_sha256"]
    require(isinstance(hashes, dict) and hashes
            and any(name.endswith(".safetensors") for name in hashes)
            and all(isinstance(name, str) and Path(name).name == name
                    and (name.endswith(".safetensors") or name == "model.safetensors.index.json")
                    and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
                    for name, digest in hashes.items()), "Invalid pinned source model-weight hashes")
    return policy


def verify_source_weights(checkpoint, policy, hash_file=sha):
    checkpoint = Path(checkpoint)
    names = {path.name for path in checkpoint.glob("*.safetensors")}
    if (checkpoint / "model.safetensors.index.json").is_file():
        names.add("model.safetensors.index.json")
    require(names == set(policy["source_weight_sha256"])
            and all(hash_file(checkpoint / name) == policy["source_weight_sha256"][name] for name in names),
            "Extension source model weights differ from the pinned policy")


def constant_schedule(steps, signature):
    return {"total_steps": steps, "peak_lr": signature["peak_lr"],
            "warmup_steps": signature["warmup_steps"],
            "decay_80_step": math.ceil(steps * .8), "decay_90_step": math.ceil(steps * .9),
            "decay_80_factor": 1.0, "decay_90_factor": 1.0, "lr_schedule": "constant"}


def verify_data_prefix(old_directory, new_directory, before, after, expected=None):
    old_directory, new_directory = Path(old_directory), Path(new_directory)
    for directory, signature in ((old_directory, before), (new_directory, after)):
        require(sha(directory / "manifest.json") == signature["data_manifest_sha256"],
                "Extension data manifest no longer matches its signature")
        for name, digest in signature["token_file_sha256"].items():
            require(sha(directory / name) == digest, f"Extension token file changed: {directory / name}")
    old_manifest, new_manifest = read(old_directory / "manifest.json"), read(new_directory / "manifest.json")
    for key in ("tokenizer", "tokenizer_revision", "block_length"):
        require(old_manifest.get(key) == new_manifest.get(key), f"Extension changes data {key}")
    require(old_manifest["splits"]["validation"] == new_manifest["splits"]["validation"],
            "Extension validation split metadata must remain identical")
    old = np.load(old_directory / "train_tokens.npy", mmap_mode="r", allow_pickle=False)
    new = np.load(new_directory / "train_tokens.npy", mmap_mode="r", allow_pickle=False)
    require(old.ndim == new.ndim == 1 and old.dtype == new.dtype == np.uint32
            and len(old) == old_manifest["splits"]["train"]["tokens"]
            and len(new) == new_manifest["splits"]["train"]["tokens"]
            and len(new) > len(old) and len(old) % before["sequence_length"] == 0,
            "Extension requires a larger uint32 dataset and no old partial block")
    original_sha, prefix_sha = token_digest(old), token_digest(new[:len(old)])
    require(original_sha == prefix_sha, "Expanded data changes the original training-token prefix")
    result = {"old_tokens": len(old), "new_tokens": len(new), "dtype": "uint32",
              "old_token_values_sha256": original_sha, "new_prefix_token_values_sha256": prefix_sha,
              "validation_file_sha256": before["token_file_sha256"]["validation_tokens.npy"]}
    require(expected is None or result == expected, "Extension data-prefix audit differs from actual bytes")
    return result


def apply_training_extension(state, resumed, signature, args, previous, dataset):
    """Publish new orders/audit only; never rewrite the source checkpoint/state."""
    from source_migration import validate_lineage

    policy = validate_policy(read(args.training_extension_file))
    checkpoint = args.resume.resolve()
    require(checkpoint == Path(policy["source_checkpoint"]) and checkpoint.name == "final_checkpoint"
            and args.data.resolve() == Path(policy["target_data"]), "Extension CLI paths differ from policy")
    require(sha(checkpoint / "checkpoint_manifest.json") == policy["source_checkpoint_manifest_sha256"]
            and sha(checkpoint / "training_state.pt") == policy["source_training_state_sha256"],
            "Extension source checkpoint differs from the pinned policy")
    verify_source_weights(checkpoint, policy)
    original, migrations = validate_lineage(state)
    before = state["signature"]
    authorize_extension_signatures(before, signature)
    old_manifest = read(checkpoint / "checkpoint_manifest.json")
    for key in ("signature", "schedule", "completed_steps", "trained_tokens", "supervised_tokens",
                "optimizer_updates", "order_metadata", "train_probe", "train_probe_history"):
        require(old_manifest.get(key) == resumed.get(key), f"Extension checkpoint/state disagree: {key}")
    require(state.get("lr_intervention") is None and state.get("lr_intervention_audit") is None,
            "Training extension cannot replace an LR intervention")
    require(state["schedule"] == constant_schedule(state["schedule"]["total_steps"], before)
            and state["completed_steps"] == state["schedule"]["total_steps"]
            and state["completed_steps"] >= before["warmup_steps"],
            "Training extension requires a completed constant-LR pilot after warmup")
    for key in ("completed_steps", "trained_tokens", "supervised_tokens", "optimizer_updates"):
        require(state[key] == policy[key], f"Extension boundary differs from policy: {key}")
    old_backend = read(checkpoint.parent / "backend_config.json")
    require(sha(checkpoint.parent / "backend_config.json") == before["backend_config_sha256"]
            and old_backend.get("execution_mode") == "standard_native",
            "Training extension requires the unchanged validated native backend")
    require(isinstance(previous, EpochOrders) and previous.epochs == 1
            and previous.metadata["generator"] != EXTENSION_GENERATOR,
            "Training extension requires an original one-epoch pilot order")
    prefix = verify_data_prefix(policy["source_data"], args.data, before, signature)
    length, batch = signature["sequence_length"], signature["effective_batch"]
    require(prefix["old_tokens"] // length == previous.blocks,
            "Original data length differs from original shuffle")
    old_counts = prefix_counts(previous, state["completed_steps"], batch, length, prefix["old_tokens"])
    require(old_counts == (state["trained_tokens"], state["supervised_tokens"], state["optimizer_updates"]),
            "Original counters do not match the completed original shuffle prefix")
    epoch_steps = math.ceil(len(dataset) / batch)
    total_steps = epoch_steps * args.epochs
    require(args.epochs == policy["target_epochs"] == 2 and total_steps == policy["target_steps"]
            and (args.steps is None or args.steps == total_steps)
            and policy["target_input_tokens"] == prefix["new_tokens"] * 2
            and total_steps > state["completed_steps"], "Extension target budget differs from policy/full two epochs")
    orders = EpochOrders.extend_completed_prefix(args.output / "orders", previous, state["completed_steps"],
                                                len(dataset), batch, args.epochs, args.seed)
    require(prefix_counts(orders, state["completed_steps"], batch, length, len(dataset.tokens)) == old_counts,
            "Expanded orders alter historical input/target/update counters")
    require(prefix_counts(orders, total_steps, batch, length, len(dataset.tokens))[0] == policy["target_input_tokens"],
            "Expanded orders do not match the exact target input budget")
    audit_policy_path = args.output / "training_extension.json"
    shutil.copy2(args.training_extension_file, audit_policy_path)
    entry = {"authorized_by": "--training-extension-file", "kind": KIND,
             "changed_fields": sorted(key for key in before if before[key] != signature[key]),
             "completed_steps": state["completed_steps"], "from_signature": before, "to_signature": signature,
             "resumed_checkpoint": str(checkpoint), "from_source_directory": str(checkpoint.parent / "source"),
             "to_source_directory": str(args.output.resolve() / "source"),
             "saved_utc": datetime.now(timezone.utc).isoformat(),
             "extension_policy": policy, "extension_policy_path": str(audit_policy_path.resolve()),
             "extension_policy_sha256": sha(audit_policy_path), "data_prefix": prefix,
             "from_schedule": dict(state["schedule"]), "to_schedule": constant_schedule(total_steps, signature),
             "from_order_metadata": previous.metadata, "to_order_metadata": orders.metadata,
             "to_order_directory": str((args.output / "orders").resolve()),
             "preserved_rng_sha256": {key: token_digest(resumed[key].numpy()) for key in ("cpu_rng", "cuda_rng")},
             "preserved_train_probe": state.get("train_probe"),
             "preservation": "Original weights, Adam tensors, RNG, probe and validation histories are loaded unchanged; only the future data/schedule budget expands."}
    state.update(signature=signature, original_signature=original, source_migrations=[*migrations, entry],
                 schedule=entry["to_schedule"], order_metadata=orders.metadata,
                 epoch_cursor={"completed_epochs": state["completed_steps"] // epoch_steps,
                               "step_in_epoch": state["completed_steps"] % epoch_steps},
                 training_extension=entry, extension_source_checkpoint=str(checkpoint))
    validate_lineage(state)
    return orders


def validate_extension_entry(entry):
    policy = validate_policy(entry.get("extension_policy"))
    before, after = entry["from_signature"], entry["to_signature"]
    authorize_extension_signatures(before, after)
    require(entry.get("kind") == KIND and entry.get("completed_steps") == policy["completed_steps"]
            and entry.get("resumed_checkpoint") == policy["source_checkpoint"], "Extension lineage boundary differs from policy")
    require(entry.get("from_schedule") == constant_schedule(policy["completed_steps"], before)
            and entry.get("to_schedule") == constant_schedule(policy["target_steps"], after)
            and policy["completed_steps"] >= before["warmup_steps"],
            "Extension changes the completed LR prefix or restarts warmup")
    metadata = entry.get("to_order_metadata", {})
    require(metadata.get("generator") == EXTENSION_GENERATOR and metadata.get("epochs") == 2
            and metadata.get("prefix_extension", {}).get("completed_steps") == policy["completed_steps"],
            "Extension lacks its preserved-prefix shuffle audit")
    require(entry.get("data_prefix", {}).get("new_tokens", 0) * 2 == policy["target_input_tokens"],
            "Extension data budget differs from its policy")
    return policy


def verify_extension_snapshots(entry, hash_file=sha):
    """CPU-only reconstruction of the immutable source, data and order proof."""
    policy = validate_extension_entry(entry)
    checkpoint = Path(policy["source_checkpoint"])
    require(hash_file(checkpoint / "checkpoint_manifest.json") == policy["source_checkpoint_manifest_sha256"]
            and hash_file(checkpoint / "training_state.pt") == policy["source_training_state_sha256"],
            "Original extension checkpoint bytes changed")
    verify_source_weights(checkpoint, policy, hash_file)
    require(hash_file(Path(entry["extension_policy_path"])) == entry["extension_policy_sha256"]
            and read(entry["extension_policy_path"]) == policy, "Extension policy snapshot changed")
    source = read(checkpoint / "checkpoint_manifest.json")
    for name, expected in (("signature", entry["from_signature"]), ("schedule", entry["from_schedule"]),
                           ("order_metadata", entry["from_order_metadata"]),
                           ("train_probe", entry["preserved_train_probe"])):
        require(source.get(name) == expected, f"Extension source snapshot differs: {name}")
    for key in ("completed_steps", "trained_tokens", "supervised_tokens", "optimizer_updates"):
        require(source.get(key) == policy[key], f"Extension source counter differs: {key}")
    before, after = entry["from_signature"], entry["to_signature"]
    backend_path = checkpoint.parent / "backend_config.json"
    require(hash_file(backend_path) == before["backend_config_sha256"]
            and read(backend_path).get("execution_mode") == "standard_native", "Extension source backend changed")
    prefix = verify_data_prefix(policy["source_data"], policy["target_data"], before, after, entry["data_prefix"])
    old_orders = EpochOrders(checkpoint / "orders", entry["from_order_metadata"])
    new_orders = EpochOrders(entry["to_order_directory"], entry["to_order_metadata"])
    count = policy["completed_steps"] * before["effective_batch"]
    require(np.array_equal(old_orders.orders[0][:count], new_orders.orders[0][:count]),
            "Extension changed a previously consumed shuffle block")
    require(new_orders.metadata["prefix_extension"]["source_order_sha256"]
            == old_orders.metadata["files"][0]["sha256"], "Extension points to a different original shuffle")
    for orders, tokens in ((old_orders, prefix["old_tokens"]), (new_orders, prefix["new_tokens"])):
        require(prefix_counts(orders, policy["completed_steps"], before["effective_batch"],
                              before["sequence_length"], tokens)
                == (policy["trained_tokens"], policy["supervised_tokens"], policy["optimizer_updates"]),
                "Extension historical counters differ from reconstructed shuffle prefix")
    require(new_orders.epoch_steps * new_orders.epochs == policy["target_steps"]
            and prefix_counts(new_orders, policy["target_steps"], after["effective_batch"],
                              after["sequence_length"], prefix["new_tokens"])[0] == policy["target_input_tokens"],
            "Extension final counters differ from its exact target budget")
    return {"status": "passed", "source_checkpoint": str(checkpoint),
            "completed_steps_preserved": policy["completed_steps"], "target_steps": policy["target_steps"],
            "target_input_tokens": policy["target_input_tokens"], "data_prefix": prefix}
