"""Independently reload and verify a train_cpt.py checkpoint on the CPU.

Usage: python verify_cpt_checkpoint.py CHECKPOINT --output RESULT.json
This validates serialization and resumable state structure; it does not compare
against the original live training tensors, execute a model forward, or run CUDA.
Large weight/state files are read for tensor checks but are not re-hashed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import traceback

from safetensors import safe_open
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from cpt_orders import EpochOrders, load_saved_orders, prefix_counts as reconstruct_prefix_counts
from source_migration import verify_lineage_snapshots
from standard_native import check_native_config
from lr_intervention import validate_intervention_audit, intervention_lr


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def integer(value, name, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, int) and value >= minimum,
            f"{name} must be an integer >= {minimum}")
    return value


def expected_schedule(total_steps, peak_lr, warmup_steps=None, lr_schedule="multistep"):
    require(lr_schedule in ("multistep", "constant"), "Unknown signed LR schedule")
    if warmup_steps is None:
        warmup_steps = max(1, math.ceil(total_steps * .05))
    integer(warmup_steps, "schedule warmup steps")
    require(warmup_steps <= total_steps, "Warmup exceeds the fixed schedule length")
    schedule = {"total_steps": total_steps, "peak_lr": peak_lr,
            "warmup_steps": warmup_steps,
            "decay_80_step": math.ceil(total_steps * .8),
            "decay_90_step": math.ceil(total_steps * .9),
            "decay_80_factor": .316, "decay_90_factor": .1}
    if lr_schedule == "constant":
        schedule.update(lr_schedule="constant", decay_80_factor=1.0, decay_90_factor=1.0)
    return schedule


def last_lr(schedule, completed_steps):
    if completed_steps == 0:
        return schedule["peak_lr"]
    step = completed_steps - 1
    if step < schedule["warmup_steps"]:
        factor = (step + 1) / schedule["warmup_steps"]
    elif step < schedule["decay_80_step"]:
        factor = 1.0
    elif step < schedule["decay_90_step"]:
        factor = schedule["decay_80_factor"]
    else:
        factor = schedule["decay_90_factor"]
    return schedule["peak_lr"] * factor


def prefix_counts(order, completed_steps, effective_batch, sequence_length, token_count):
    """Reconstruct counters and updates, including a one-token, zero-target tail."""
    return reconstruct_prefix_counts(order, completed_steps, effective_batch, sequence_length, token_count)


def verify_signature_and_counters(checkpoint, state, manifest):
    require(state.get("schema_version") == 1, "Expected train_cpt.py training-state schema_version=1")
    signature = state.get("signature")
    require(isinstance(signature, dict), "Missing training signature")
    schedule = state.get("schedule")
    require(isinstance(schedule, dict), "Missing fixed learning-rate schedule")
    total_steps = integer(schedule.get("total_steps"), "schedule.total_steps", 1)
    peak_lr = schedule.get("peak_lr")
    require(isinstance(peak_lr, (float, int)) and not isinstance(peak_lr, bool)
            and math.isfinite(peak_lr) and peak_lr > 0, "Invalid schedule peak learning rate")
    lr_schedule = signature.get("lr_schedule", "multistep")
    warmup_steps = signature.get("warmup_steps")
    if "warmup_steps" in signature:
        integer(warmup_steps, "signature.warmup_steps")
    if "lr_schedule" in signature:
        require(lr_schedule == "constant", "Only nondefault LR schedules add a signature field")
    require(schedule == expected_schedule(total_steps, peak_lr, warmup_steps, lr_schedule),
            "Inconsistent fixed LR schedule or signed warmup/schedule policy")
    validate_intervention_audit(state)
    require(signature.get("peak_lr") == peak_lr, "Signature LR differs from the saved schedule")
    if state.get("lr_intervention") is not None:
        require(state["lr_intervention"]["start_lr"] == peak_lr
                and state["lr_intervention"]["end_lr"] <= peak_lr,
                "Saved LR intervention differs from the original peak LR")
    integer(signature.get("seed"), "signature.seed", -(2 ** 63))
    length = integer(signature.get("sequence_length"), "signature.sequence_length", 2)
    effective_batch = integer(signature.get("effective_batch"), "signature.effective_batch", 1)
    completed = integer(state.get("completed_steps"), "completed_steps")
    updates = integer(state.get("optimizer_updates"), "optimizer_updates")
    require(updates <= completed <= total_steps, "Update counters exceed the fixed schedule")
    if checkpoint.name == "final_checkpoint":
        require(completed == total_steps, "final_checkpoint does not complete its fixed schedule")
    if checkpoint.name == "stopped_checkpoint":
        require(completed < total_steps, "stopped_checkpoint already completed its schedule")
    order = load_saved_orders(state, checkpoint)
    blocks = order.blocks if isinstance(order, EpochOrders) else len(order)
    epochs = integer(signature.get("epochs", 1), "signature.epochs", 1)
    epoch_steps = math.ceil(blocks / effective_batch)
    if isinstance(order, EpochOrders):
        require(order.epochs == epochs and order.batch == effective_batch
                and order.metadata.get("seed") == signature["seed"], "Saved compact order metadata differs from signature")
        require(manifest.get("order_metadata") == state["order_metadata"], "Checkpoint manifest shuffle metadata differs")
    else:
        require(epochs == 1, "Historical shuffle lists only support one epoch")
    require(total_steps <= epoch_steps * epochs, "Schedule exceeds configured dataset epochs")
    if "epoch_cursor" in state:
        require(state["epoch_cursor"] == {"completed_epochs": completed // epoch_steps,
                                          "step_in_epoch": completed % epoch_steps}, "Saved epoch cursor differs from global step")
        require(manifest.get("epoch_cursor") == state["epoch_cursor"], "Manifest/state epoch cursors differ")
    for name in ("trained_tokens", "supervised_tokens"):
        integer(state.get(name), name)
    require(state["supervised_tokens"] <= state["trained_tokens"], "Targets exceed input tokens")
    for name in ("cpu_rng", "cuda_rng"):
        value = state.get(name)
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                f"Invalid saved {name}")
    torch.Generator(device="cpu").set_state(state["cpu_rng"])
    source_hashes = signature.get("source_sha256")
    require(isinstance(source_hashes, dict) and len(source_hashes) > 0, "Missing training-source hashes")
    for name, digest in source_hashes.items():
        require(Path(name).name == name and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid source hash entry: {name}")
        source = checkpoint.parent / "source" / name
        require(source.is_file() and sha(source) == digest, f"Training-source snapshot mismatch: {name}")
    for key in ("backend_config_sha256", "data_manifest_sha256"):
        require(isinstance(signature.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", signature[key]) is not None,
                f"Invalid signature hash: {key}")
    backend_path = checkpoint.parent / "backend_config.json"
    data_path = checkpoint.parent / "data_manifest.json"
    require(backend_path.is_file() and sha(backend_path) == signature["backend_config_sha256"],
            "Backend-config snapshot/hash mismatch")
    require(data_path.is_file() and sha(data_path) == signature["data_manifest_sha256"],
            "Data-manifest snapshot/hash mismatch")
    backend, data = read_json(backend_path), read_json(data_path)
    require(backend.get("validated") is True and backend.get("schema_version") == 1,
            "Saved backend was not validated")
    require(backend.get("sequence_length") == length and backend.get("effective_batch") == effective_batch,
            "Backend-config batch/context differs from the signature")
    require(data.get("tokenizer") == backend.get("model")
            and data.get("tokenizer_revision") == backend.get("revision"), "Tokenizer data/model revision mismatch")
    token_hashes = signature.get("token_file_sha256")
    require(isinstance(token_hashes, dict), "Missing token-file hashes")
    for name in ("train_tokens.npy", "validation_tokens.npy"):
        digest = token_hashes.get(name)
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Invalid saved token-file hash: {name}")
        documented = data.get("artifacts", {}).get(name, {}).get("sha256")
        require(documented is None or documented == digest, f"Token manifest hash differs: {name}")
    if backend.get("execution_mode") == "standard_native":
        check_native_config(backend)
    token_count = integer(data.get("splits", {}).get("train", {}).get("tokens"), "dataset train tokens", 1)
    require(blocks == math.ceil(token_count / length), "Shuffle length differs from token-budget block count")
    inputs, targets, applied = prefix_counts(order, completed, effective_batch, length, token_count)
    require((state["trained_tokens"], state["supervised_tokens"], updates) == (inputs, targets, applied),
            "Saved input/target/optimizer counters differ from the completed shuffle prefix")
    last_update_input_tokens = prefix_counts(order, max(0, completed - 1), effective_batch, length, token_count)[0]
    audit = state.get("lr_intervention_audit")
    if audit is not None:
        audit_inputs = prefix_counts(order, audit["applied_at_completed_steps"], effective_batch, length, token_count)[0]
        require(audit_inputs == audit["applied_at_input_tokens"], "LR intervention audit differs from the saved shuffle prefix")
        require(state["lr_intervention"]["end_input_tokens"] <= prefix_counts(order, total_steps, effective_batch, length, token_count)[0],
                "LR intervention exceeds the configured token budget")
    for key in ("completed_steps", "optimizer_updates", "trained_tokens", "supervised_tokens", "signature", "schedule"):
        require(manifest.get(key) == state.get(key), f"Checkpoint manifest and state disagree: {key}")
    for key in ("lr_intervention", "lr_intervention_audit"):
        require(manifest.get(key) == state.get(key), f"Checkpoint manifest/state LR intervention differs: {key}")
    lineage = verify_lineage_snapshots(state, sha)
    for key, fallback in (("original_signature", signature), ("source_migrations", []), ("native_retry_count", 0)):
        require(manifest.get(key, fallback) == state.get(key, fallback),
                f"Checkpoint manifest/state source-migration or retry audit differs: {key}")
    for key in ("training_extension", "extension_source_checkpoint"):
        require(manifest.get(key) == state.get(key), f"Checkpoint manifest/state training extension differs: {key}")
    extensions = [entry for entry in state.get("source_migrations", [])
                  if entry.get("authorized_by") == "--training-extension-file"]
    if extensions:
        require(len(extensions) == 1 and state.get("training_extension") == extensions[0]
                and state.get("extension_source_checkpoint") == extensions[0]["resumed_checkpoint"],
                "Training extension metadata differs from the signed lineage")
        require(state.get("train_probe") == extensions[0]["preserved_train_probe"],
                "Extended training changed the original fixed training probe")
    else:
        require(state.get("training_extension") is None and state.get("extension_source_checkpoint") is None,
                "Training extension metadata lacks a signed lineage")
    parent_manifest = checkpoint.parent / "manifest.json"
    if parent_manifest.is_file():
        parent = read_json(parent_manifest)
        require(parent.get("signature") == signature, "Run manifest signature differs from checkpoint")
        arguments = parent.get("arguments", {})
        require(arguments.get("lr_schedule", "multistep") == lr_schedule
                and arguments.get("warmup_steps") == warmup_steps,
                "Run CLI warmup/LR schedule differs from checkpoint signature")
        require(parent.get("original_signature", signature) == lineage["original_signature"]
                and parent.get("source_migrations", []) == lineage["source_migrations"],
                "Run manifest source-migration audit differs from checkpoint")
        for key in ("lr_intervention", "lr_intervention_audit"):
            require(parent.get(key) == state.get(key), f"Run manifest/checkpoint LR intervention differs: {key}")
        for key in ("training_extension", "extension_source_checkpoint"):
            require(parent.get(key) == state.get(key), f"Run manifest/checkpoint training extension differs: {key}")
    probe = state.get("train_probe")
    require(manifest.get("train_probe") == probe, "Checkpoint/state fixed train probe differs")
    require(manifest.get("train_probe_history", []) == state.get("train_probe_history", []), "Checkpoint/state probe history differs")
    if probe is not None:
        count = integer(probe.get("blocks"), "probe.blocks", 1)
        integer(probe.get("seed"), "probe.seed", -(2 ** 63))
        indices = probe.get("indices")
        require(isinstance(indices, list) and len(indices) == count
                and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < token_count // length for index in indices)
                and indices == sorted(set(indices)), "Invalid fixed complete-block train probe")
        values = np.asarray(indices, dtype=np.int64)
        require(hashlib.sha256(memoryview(values).cast("B")).hexdigest() == probe.get("indices_sha256"), "Fixed probe indices checksum differs")
        require(isinstance(probe.get("tokens_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", probe["tokens_sha256"]), "Invalid fixed probe token checksum")
        require((probe.get("input_tokens"), probe.get("supervised_tokens")) == (count * length, count * (length - 1)), "Fixed probe counts differ")
    return {"execution_mode": backend.get("execution_mode", "configured"), "compiled_optimizer": backend.get("compiled_optimizer", False),
            "completed_steps": completed, "configured_steps": total_steps, "optimizer_updates": updates,
            "trained_tokens": inputs, "supervised_tokens": targets, "dataset_tokens": token_count,
            "shuffle_blocks": blocks, "epochs": epochs, "epoch_steps": epoch_steps,
            "order_storage": "checksummed mmap NumPy files" if isinstance(order, EpochOrders) else "legacy Python list",
            "schedule_complete": completed == total_steps,
            "schedule": schedule, "last_update_input_tokens": last_update_input_tokens,
            "lr_intervention": state.get("lr_intervention"), "lr_intervention_audit": audit,
            "source_sha256": source_hashes, "signature": signature, **lineage}


def verify_weights(checkpoint, model):
    loaded = model.state_dict()
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        require(isinstance(weight_map, dict) and len(weight_map) > 0, "Invalid safetensors weight index")
    else:
        weight_path = checkpoint / "model.safetensors"
        require(weight_path.is_file(), "Missing indexed or single-file safetensors model")
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            weight_map = dict.fromkeys(handle.keys(), weight_path.name)
        index = {}
    require(set(weight_map) <= set(loaded), "Serialized tensors are absent from the reloaded model")
    compared, elements, bytes_checked = 0, 0, 0
    for filename in sorted(set(weight_map.values())):
        require(Path(filename).name == filename, "Invalid safetensors shard path")
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            expected_keys = {name for name, shard in weight_map.items() if shard == filename}
            require(set(handle.keys()) == expected_keys, f"Shard contents disagree with its weight index: {filename}")
            for name in sorted(expected_keys):
                tensor, restored = handle.get_tensor(name), loaded[name]
                require(tensor.dtype == torch.float32 and restored.dtype == torch.float32,
                        f"Weight precision is not FP32: {name}")
                require(tensor.shape == restored.shape and torch.equal(tensor, restored),
                        f"Serialized weight differs from model reload: {name}")
                require(bool(torch.isfinite(tensor).all()), f"Nonfinite model weight: {name}")
                compared += 1
                elements += tensor.numel()
                bytes_checked += tensor.numel() * tensor.element_size()
    aliases = {}
    for name in sorted(set(loaded) - set(weight_map)):
        tensor = loaded[name]
        matches = [other for other in weight_map if tensor.data_ptr() == loaded[other].data_ptr()
                   and tensor.shape == loaded[other].shape and tensor.dtype == loaded[other].dtype]
        require(matches and model.config.tie_word_embeddings and name in ("lm_head.weight", "model.embed_tokens.weight"),
                f"Reloaded tensor missing from safetensors without an allowed tied-weight alias: {name}")
        aliases[name] = matches[0]
    if "total_size" in index.get("metadata", {}):
        require(index["metadata"]["total_size"] == bytes_checked, "Safetensors total_size metadata mismatch")
    require(all(parameter.dtype == torch.float32 for parameter in model.parameters()), "Reloaded parameter dtype mismatch")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if "total_parameters" in index.get("metadata", {}):
        require(index["metadata"]["total_parameters"] == parameter_count,
                "Safetensors total_parameters metadata mismatch")
    return {"serialized_tensors_compared_exactly": compared, "reloaded_state_tensors": len(loaded),
            "serialized_tensor_elements": elements, "serialized_tensor_bytes": bytes_checked,
            "model_parameters": parameter_count,
            "unique_parameter_tensors": len(list(model.parameters())), "tied_aliases": aliases,
            "shards": len(set(weight_map.values()))}


def verify_optimizer(state, model, counters):
    optimizer = state.get("optimizer")
    require(isinstance(optimizer, dict), "Missing serialized optimizer")
    groups, states = optimizer.get("param_groups"), optimizer.get("state")
    require(isinstance(groups, list) and len(groups) > 0 and isinstance(states, dict), "Invalid optimizer state dictionary")
    parameters = list(model.parameters())
    ids = [index for group in groups for index in group["params"]]
    require(len(ids) == len(parameters) and len(set(ids)) == len(ids), "Optimizer groups do not cover each model parameter once")
    require(set(states) <= set(ids), "Optimizer contains an unknown parameter state")
    updates = counters["optimizer_updates"]
    require(len(states) == len(parameters) if updates else len(states) == 0,
            "Initialized optimizer state count differs from actual applied updates")
    tensor_count, moment_elements, moment_bytes = 0, 0, 0
    for group in groups:
        require(tuple(group.get("betas", ())) == (.9, .95) and group.get("weight_decay") == .1,
                "AdamW betas or weight decay differ from the CPT protocol")
        require(not group.get("amsgrad", False), "Expected the CPT protocol's ordinary AdamW moments")
        mathematical_lr = last_lr(state["schedule"], counters["completed_steps"])
        policy = state.get("lr_intervention")
        if policy is not None and counters["last_update_input_tokens"] >= policy["start_input_tokens"]:
            mathematical_lr = intervention_lr(policy, counters["last_update_input_tokens"])
        native = counters.get("execution_mode") == "standard_native"
        expected_lr = float(np.float32(mathematical_lr)) if native or counters.get("compiled_optimizer", False) else mathematical_lr
        require(float(group.get("lr")) == expected_lr,
                "Saved optimizer LR differs from the exact implementation precision at completed schedule boundary")
        require(group.get("eps") == 1e-8 and not group.get("maximize", False) and not group.get("differentiable", False),
                "AdamW epsilon/maximize/differentiable differs from CPT math protocol")
        if native:
            require(isinstance(group.get("lr"), float) and not group.get("capturable", False)
                    and group.get("foreach") is True and group.get("fused") is False,
                    "Standard native optimizer flags/LR representation differ")
    for index, parameter in zip(ids, parameters):
        if index not in states:
            continue
        values = states[index]
        require(set(values) == {"step", "exp_avg", "exp_avg_sq"},
                f"Unexpected AdamW state entries for parameter {index}")
        step = values.get("step")
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and step.dtype == torch.float32
                and step.device.type == "cpu" and bool(torch.isfinite(step).all()),
                f"Invalid optimizer step for parameter {index}")
        require(float(step) == updates, f"Optimizer step differs from optimizer_updates for parameter {index}")
        tensor_count += 1
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = values.get(name)
            require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                    and tensor.device.type == "cpu" and tensor.shape == parameter.shape,
                    f"Invalid FP32 optimizer {name} for parameter {index}")
            require(bool(torch.isfinite(tensor).all()), f"Nonfinite optimizer {name} for parameter {index}")
            if name == "exp_avg_sq":
                require(bool((tensor >= 0).all()), f"Negative AdamW second moment for parameter {index}")
            tensor_count += 1
            moment_elements += tensor.numel()
            moment_bytes += tensor.numel() * tensor.element_size()
    # Bind the serialized states to an actual CPU AdamW as an additional shape /
    # group compatibility check. This performs no optimizer update.
    position, bound_groups = 0, []
    for group in groups:
        count = len(group["params"])
        bound_groups.append({"params": parameters[position:position + count]})
        position += count
    restored = torch.optim.AdamW(bound_groups, lr=state["schedule"]["peak_lr"], betas=(.9, .95), weight_decay=.1)
    restored.load_state_dict(optimizer)
    require(len(restored.state) == len(states), "AdamW reload lost optimizer states")
    for value in restored.state.values():
        require(value["exp_avg"].dtype == torch.float32 and value["exp_avg_sq"].dtype == torch.float32,
                "AdamW reload changed moment dtype")
    return {"parameter_groups": len(groups), "parameter_ids": len(ids), "initialized_states": len(states),
            "state_tensors_checked": tensor_count, "moment_elements": moment_elements,
            "moment_bytes": moment_bytes, "moment_dtype": "float32", "optimizer_step": updates,
            "cpu_adamw_load_state_dict_passed": True}


def verify_tokenizer(model, tokenizer):
    vocabulary = integer(model.config.vocab_size, "model vocabulary", 1)
    eos = integer(tokenizer.eos_token_id, "tokenizer EOS token")
    require(eos < vocabulary and len(tokenizer) <= vocabulary, "Tokenizer vocabulary/EOS exceeds model embedding vocabulary")
    ids = list(tokenizer.get_vocab().values())
    require(ids and min(ids) >= 0 and max(ids) < vocabulary, "Tokenizer token IDs exceed model vocabulary")
    model_eos = model.config.eos_token_id
    require(eos in model_eos if isinstance(model_eos, list) else eos == model_eos, "Model/tokenizer EOS differs")
    require(tokenizer.eos_token is not None and tokenizer.encode(tokenizer.eos_token, add_special_tokens=False) == [eos],
            "EOS does not tokenize to its saved token ID")
    inputs, outputs = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    require(inputs.shape[0] == outputs.shape[0] == vocabulary, "Model embedding / output vocabulary differs from config")
    if model.config.tie_word_embeddings:
        require(inputs.data_ptr() == outputs.data_ptr(), "Model embedding/output weights were not tied on reload")
    return {"model_vocabulary": vocabulary, "tokenizer_size": len(tokenizer), "eos_token_id": eos,
            "eos_token": tokenizer.eos_token, "tied_word_embeddings": model.config.tie_word_embeddings}


def verify(checkpoint):
    require(checkpoint.is_dir(), f"Checkpoint directory does not exist: {checkpoint}")
    require(not checkpoint.name.endswith(".incomplete"), "Refusing an unpublished .incomplete checkpoint")
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    require(manifest.get("schema_version") == 1, "Unsupported checkpoint manifest schema")
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    counters = verify_signature_and_counters(checkpoint, state, manifest)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
                                                torch_dtype=torch.float32, device_map=None)
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "Model reload left the CPU")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    weights = verify_weights(checkpoint, model)
    optimizer = verify_optimizer(state, model, counters)
    files = {path.name: path.stat().st_size for path in checkpoint.iterdir() if path.is_file()}
    order_files = ({str(path.relative_to(checkpoint)): path.stat().st_size for path in (checkpoint / "orders").iterdir() if path.is_file()}
                   if (checkpoint / "orders").is_dir() else {})
    for name, size in manifest.get("files", {}).items():
        require(files.get(name) == size, f"Saved checkpoint file size differs: {name}")
    return {"status": "passed", "checkpoint": str(checkpoint), "cpu_only": True,
            "weights": weights, "optimizer": optimizer, "tokenizer": verify_tokenizer(model, tokenizer),
            "training": counters, "checkpoint_bytes": sum(files.values()) + sum(order_files.values()),
            "checkpoint_files": files, "checkpoint_order_files": order_files,
            "checkpoint_config_sha256": sha(checkpoint / "config.json"),
            "checkpoint_manifest_sha256": sha(checkpoint / "checkpoint_manifest.json"),
            "checkpoint_source_sha256": counters["source_sha256"],
            "scope": "Exact serialized weight reload; FP32 finite weights/moments; optimizer binding/counters; source/config/data signatures. No live-model comparison or training resume execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    require(args.threads > 0, "--threads must be positive")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    try:
        result = verify(args.checkpoint.resolve())
    except Exception as error:
        result = {"status": "failed", "checkpoint": str(args.checkpoint.resolve()),
                  "error": str(error), "traceback": traceback.format_exc(), "cpu_only": True}
    result.update(seconds=time.perf_counter() - started, verified_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_source_sha256=sha(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "seconds": result["seconds"], "error": result.get("error")}), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
