"""Train or resume one CPT epoch using a separately validated local backend.

Forward uses BF16 autocast; master parameters, gradients and AdamW moments stay
FP32. A stop boundary changes execution length, never the saved LR schedule.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import torch
from transformers import AutoTokenizer

import cpt
from compiled_adamw import make_compiled_adamw, set_lr
from model_backends import configure_model, eager_native
from packed_data import PackedTokens
from token_batches import PackedTokenBatches


SCHEMA_VERSION = 1
SOURCE_FILES = ("train_cpt.py", "model_backends.py", "token_batches.py", "compiled_adamw.py",
                "native_ops.py", "native_chunked_ce.py", "cpt.py", "packed_data.py")


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def load_backend(path):
    config = json.loads(path.read_text())
    required = {"schema_version", "validated", "model", "revision", "sequence_length",
                "effective_batch", "micro_batch", "mask_policy", "fused_optimizer", "options"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Backend configuration is missing {sorted(missing)}")
    if config["schema_version"] != SCHEMA_VERSION or config["validated"] is not True:
        raise ValueError("Backend configuration must have schema_version=1 and validated=true")
    for name in ("sequence_length", "effective_batch", "micro_batch"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Backend {name} must be a positive integer")
    if config["sequence_length"] < 2 or config["micro_batch"] > config["effective_batch"]:
        raise ValueError("Sequence length must be >=2 and micro batch cannot exceed effective batch")
    if config["mask_policy"] not in ("ones", "causal") or not isinstance(config["fused_optimizer"], bool):
        raise ValueError("Expected mask_policy=ones/causal and boolean fused_optimizer")
    compiled_optimizer = config.get("compiled_optimizer", False)
    if not isinstance(compiled_optimizer, bool):
        raise ValueError("Expected boolean compiled_optimizer when provided")
    if compiled_optimizer and config["fused_optimizer"]:
        raise ValueError("compiled_optimizer and fused_optimizer are mutually exclusive")
    if not isinstance(config["options"], dict):
        raise ValueError("Backend options must be a dictionary")
    options = dict(config["options"])
    expected_shape = [config["micro_batch"], config["sequence_length"]]
    if "compile_shape" in options and options["compile_shape"] != expected_shape:
        raise ValueError("Backend compile_shape differs from the validated micro batch/context")
    options["compile_shape"] = expected_shape
    config["options"] = options
    return config


def make_schedule(total_steps, peak_lr):
    return {"total_steps": total_steps, "peak_lr": peak_lr,
            "warmup_steps": max(1, math.ceil(total_steps * .05)),
            "decay_80_step": math.ceil(total_steps * .8),
            "decay_90_step": math.ceil(total_steps * .9),
            "decay_80_factor": .316, "decay_90_factor": .1}


def learning_rate(schedule, step):
    """LR for a zero-based data update, identical before and after resuming."""
    if not 0 <= step < schedule["total_steps"]:
        raise ValueError("Step lies outside the saved schedule")
    if step < schedule["warmup_steps"]:
        factor = (step + 1) / schedule["warmup_steps"]
    elif step < schedule["decay_80_step"]:
        factor = 1.0
    elif step < schedule["decay_90_step"]:
        factor = schedule["decay_80_factor"]
    else:
        factor = schedule["decay_90_factor"]
    return schedule["peak_lr"] * factor


def selected_token_counts(dataset, order, total_steps, effective_batch):
    """Count actual inputs/targets, including the one possible right-padded tail."""
    selection = order[:min(len(order), total_steps * effective_batch)]
    full = dataset.sequence_length
    remainder = len(dataset.tokens) % full
    inputs = len(selection) * full
    if remainder and len(dataset) - 1 in selection:
        inputs -= full - remainder
    return inputs, inputs - len(selection)


def validate_order(order, dataset_length):
    if not isinstance(order, list) or len(order) != dataset_length:
        raise ValueError("Saved shuffle order does not cover this dataset")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in order):
        raise ValueError("Saved shuffle order contains noninteger indices")
    if sorted(order) != list(range(dataset_length)):
        raise ValueError("Saved shuffle order is not a dataset permutation")


def validate_resume(state, signature, dataset, args):
    if state.get("schema_version") != SCHEMA_VERSION or state.get("signature") != signature:
        raise ValueError("Resume configuration/data/LR/seed differs from the saved training state")
    validate_order(state["order"], len(dataset))
    schedule = state["schedule"]
    epoch_steps = math.ceil(len(dataset) / signature["effective_batch"])
    if not 1 <= schedule["total_steps"] <= epoch_steps:
        raise ValueError("Saved schedule is not a one-epoch step budget")
    if args.steps is not None and min(args.steps, epoch_steps) != schedule["total_steps"]:
        raise ValueError("--steps cannot change a resumed LR schedule")
    if schedule != make_schedule(schedule["total_steps"], args.lr):
        raise ValueError("Saved learning-rate schedule is inconsistent")
    step = state["completed_steps"]
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < schedule["total_steps"]:
        raise ValueError("Resume requires an incomplete checkpoint")
    inputs, targets = selected_token_counts(dataset, state["order"], step, signature["effective_batch"])
    if state["trained_tokens"] != inputs or state["supervised_tokens"] != targets:
        raise ValueError("Saved counters do not match the completed shuffle-order prefix")


def verify_data(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("tokenizer") != config["model"] or manifest.get("tokenizer_revision") != config["revision"]:
        raise ValueError("Prepared-data tokenizer does not match the selected base model/revision")
    # This trainer owns the block layout: it never passes reset position IDs,
    # document isolation masks, or leading/interior padding to the model.
    if manifest.get("packing") != "document text + EOS, concatenated within each split; no cross-document attention mask":
        raise ValueError("This trainer requires continuous-position, concatenated causal token blocks")
    hashes = {}
    for name in ("train_tokens.npy", "validation_tokens.npy"):
        hashes[name] = cpt.sha(directory / name)
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if expected is not None and hashes[name] != expected:
            raise ValueError(f"Prepared token file checksum mismatch: {name}")
    return manifest, hashes


def check_model_safety(model):
    if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
        raise ValueError("Selected backend requires Qwen2 full causal attention")
    if any(layer.attention_type != "full_attention" for layer in model.model.layers):
        raise ValueError("Resolved causal mask cannot be used for sliding/other attention layers")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("Expected full-parameter training with FP32 master parameters")


def check_optimizer_precision(model, optimizer):
    if not all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
               for p in model.parameters()):
        raise RuntimeError("Master parameter/gradient precision changed")
    for state in optimizer.state.values():
        if state["exp_avg"].dtype != torch.float32 or state["exp_avg_sq"].dtype != torch.float32:
            raise RuntimeError("AdamW moments must remain FP32")


def restore_optimizer(optimizer, saved):
    # Checkpoint tensors were loaded on CPU. Stage the large moment tensors
    # through pinned memory just as cpt.load_model does for master weights;
    # keep foreach Adam's scalar step on CPU unless the optimizer requires GPU.
    step_on_gpu = any(group.get("capturable", False) or group.get("fused", False)
                      for group in optimizer.param_groups)
    for state in saved["state"].values():
        for name, tensor in list(state.items()):
            if isinstance(tensor, torch.Tensor) and (name != "step" or step_on_gpu):
                state[name] = tensor.pin_memory().to("cuda", non_blocking=True)
    # load_state_dict replaces the constructor's parameter-group values.  A
    # saved tensor LR was mapped to CPU alongside the moments and must return
    # to CUDA before a capturable optimizer can read/update it in place.
    if step_on_gpu:
        for group in saved["param_groups"]:
            if isinstance(group.get("lr"), torch.Tensor):
                group["lr"] = group["lr"].to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    optimizer.load_state_dict(saved)


@torch.no_grad()
def evaluate(model, dataset, provider, micro_batch=4):
    """Native, full-block held-out text loss; never compile a validation graph."""
    was_training = model.training
    model.eval()
    total = torch.zeros((), device="cuda")
    count = 0
    try:
        with eager_native(model):
            for start in range(0, len(dataset), micro_batch):
                values, targets, _ = provider(dataset, range(start, min(start + micro_batch, len(dataset))))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**values, use_cache=False).loss
                total += loss * targets
                count += targets
        torch.cuda.synchronize()
        if count == 0:
            raise ValueError("Validation contains no complete supervised token block")
        loss = float(total / count)
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite native validation loss")
        return {"loss": loss, "perplexity": math.exp(loss), "supervised_tokens": count,
                "input_tokens": len(dataset) * dataset.sequence_length, "blocks": len(dataset),
                "loss_backend": "native eager", "attention_mask": "cached all-ones tensor"}
    finally:
        model.train(was_training)


def update(model, forward, optimizer, dataset, provider, indices, micro_batch, mask_policy,
           step_callable=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [provider(dataset, indices[start:start + micro_batch])
               for start in range(0, len(indices), micro_batch)]
    targets = sum(n for _, n, _ in batches)
    inputs = sum(n for _, _, n in batches)
    loss_sum = torch.zeros((), device="cuda")
    # A tail of one real token has no next-token label. Count its input while
    # skipping an all-ignored microbatch, rather than asking CE for a NaN mean.
    if targets == 0:
        return loss_sum, loss_sum.clone(), inputs, targets, False
    for values, count, _ in batches:
        if count == 0:
            continue
        if mask_policy == "causal":
            values["attention_mask"] = {"full_attention": None}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = forward(**values, use_cache=False).loss
        weighted = loss * (count / targets)
        weighted.backward()
        loss_sum += weighted.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(loss_sum):
        raise RuntimeError("Nonfinite training loss")
    if step_callable is None:
        optimizer.step()
    else:
        step_callable()
    return loss_sum.detach(), norm.detach(), inputs, targets, True


def save_checkpoint(path, model, tokenizer, optimizer, state, metadata):
    """Publish a complete checkpoint directory via one same-filesystem rename."""
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint output already exists: {path}")
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    payload = dict(state, schema_version=SCHEMA_VERSION, optimizer=optimizer.state_dict(),
                   cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    torch.save(payload, temporary / "training_state.pt")
    cpt.write(temporary / "checkpoint_manifest.json", {
        "schema_version": SCHEMA_VERSION, "completed_steps": state["completed_steps"],
        "optimizer_updates": state["optimizer_updates"], "trained_tokens": state["trained_tokens"],
        "supervised_tokens": state["supervised_tokens"], "signature": state["signature"],
        "schedule": state["schedule"], "backend": metadata,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "files": {item.name: item.stat().st_size for item in temporary.iterdir() if item.is_file()},
        "verification": "Atomic complete-directory publication; full tensor comparison is a separate diagnostic",
    })
    temporary.rename(path)
    return path.resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend-config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=positive_integer, help="Cap the fixed schedule to at most one epoch")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=positive_integer, default=20)
    p.add_argument("--checkpoint-every", type=positive_integer)
    p.add_argument("--resume", type=Path, help="Incomplete checkpoint directory; use a new output directory")
    p.add_argument("--stop-after", type=positive_integer, help="Stop at this global step without shortening the schedule")
    p.add_argument("--max-seconds", type=float, help="Wall time checked at update boundaries; saving can extend beyond it")
    return p


def main():
    started_wall = time.perf_counter()
    args = parser().parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    config = load_backend(args.backend_config)
    data_manifest, data_hashes = verify_data(args.data, config)
    config_hash = cpt.sha(args.backend_config)
    signature = {"backend_config_sha256": config_hash,
                 "data_manifest_sha256": cpt.sha(args.data / "manifest.json"),
                 "token_file_sha256": data_hashes, "peak_lr": args.lr, "seed": args.seed,
                 "sequence_length": config["sequence_length"], "effective_batch": config["effective_batch"]}
    source_directory = Path(__file__).resolve().parent
    source_hashes = {name: cpt.sha(source_directory / name) for name in SOURCE_FILES}
    # A resumed run must use exactly the same training/data/backend source.
    # Changing logging frequency or checkpoint cadence remains safe.
    signature["source_sha256"] = source_hashes
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output / "source"
    snapshots.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(source_directory / name, snapshots / name)
    shutil.copy2(args.backend_config, args.output / "backend_config.json")
    shutil.copy2(args.data / "manifest.json", args.output / "data_manifest.json")
    manifest = {"schema_version": SCHEMA_VERSION, "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "backend_config": config, "signature": signature, "source_sha256": source_hashes,
                "precision": "BF16 autocast; FP32 parameters, gradients and AdamW moments",
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "liger-kernel", "numpy")},
                "timing": "Training-loop totals include data/H2D, synchronization and logging; exclude validation/checkpoint sections. Per-record seconds time the update itself.",
                "stop_boundary": "Wall-clock limits are checked between updates; final validation/save may extend beyond the limit"}
    cpt.write(args.output / "manifest.json", manifest)
    telemetry = cpt.Telemetry(args.output)
    telemetry.thread.start()
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA GPU with BF16 support is required")
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        tokenizer_source = str(args.resume) if args.resume is not None else config["model"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=None if args.resume else config["revision"],
                                                  local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("A tokenizer EOS token is required for trailing block padding")
        train = PackedTokens(args.data, sequence_length=config["sequence_length"], include_tail=True,
                             pad_token_id=tokenizer.eos_token_id)
        validation = PackedTokens(args.data, "validation", sequence_length=config["sequence_length"])
        if len(train) == 0 or len(validation) == 0:
            raise ValueError("Training and complete-block validation data must be nonempty")
        train.explicit_mask = config["mask_policy"] == "ones"
        validation.explicit_mask = True
        provider = PackedTokenBatches()
        if args.resume is not None:
            resumed = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
            validate_resume(resumed, signature, train, args)
            state = {key: value for key, value in resumed.items()
                     if key not in ("optimizer", "cpu_rng", "cuda_rng", "schema_version")}
            # Periodic checkpoints capture an invocation before its final
            # boundary is written.  Close that history at the resumed prefix,
            # before appending this process's new invocation below.
            if state.get("invocations") and "end_step" not in state["invocations"][-1]:
                state["invocations"][-1]["end_step"] = state["completed_steps"]
        else:
            total_steps = math.ceil(len(train) / config["effective_batch"])
            if args.steps is not None:
                total_steps = min(args.steps, total_steps)
            state = {"signature": signature, "order": torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed)).tolist(),
                     "schedule": make_schedule(total_steps, args.lr), "completed_steps": 0,
                     "optimizer_updates": 0, "trained_tokens": 0, "supervised_tokens": 0,
                     "train_seconds": 0., "steady_seconds": 0., "steady_tokens": 0,
                     "validation_history": [], "validation_before": None,
                     "invocations": []}
        schedule = state["schedule"]
        if args.stop_after is not None and args.stop_after <= state["completed_steps"]:
            raise ValueError("--stop-after must exceed the resumed completed-step boundary")
        expected_inputs, expected_targets = selected_token_counts(train, state["order"], schedule["total_steps"], config["effective_batch"])
        manifest.update(schedule=schedule, configured_input_tokens=expected_inputs,
                        configured_supervised_tokens=expected_targets, dataset_input_tokens=len(train.tokens),
                        resumed_completed_steps=state["completed_steps"])
        cpt.write(args.output / "manifest.json", manifest)
        model_args = argparse.Namespace(model=str(args.resume) if args.resume else config["model"],
                                       revision=None if args.resume else config["revision"])
        model = cpt.load_model(model_args)
        check_model_safety(model)
        forward, backend_metadata = configure_model(model, config["options"])
        optimizer_compile = None
        step_callable = None
        if config.get("compiled_optimizer", False):
            optimizer, step_callable, optimizer_compile = make_compiled_adamw(model.parameters(), args.lr)
        else:
            optimizer = cpt.make_optimizer(model, args.lr, config["fused_optimizer"])
        backend_metadata["optimizer_compile"] = optimizer_compile
        state["optimizer_compile"] = optimizer_compile
        if args.resume is not None:
            restore_optimizer(optimizer, resumed["optimizer"])
            check_optimizer_precision(model, optimizer)
            torch.set_rng_state(resumed["cpu_rng"])
            torch.cuda.set_rng_state(resumed["cuda_rng"])
            del resumed
            gc.collect()
        telemetry.phase = "validation_before"
        before = evaluate(model, validation, provider)
        if state["validation_before"] is None:
            state["validation_before"] = before
        cpt.write(args.output / "validation_before.json", dict(step=state["completed_steps"], **before))
        print(json.dumps({"event": "validation_before", "step": state["completed_steps"], **before}), flush=True)
        invocation = {"output": str(args.output.resolve()), "start_step": state["completed_steps"],
                      "first_full_shape_update_seconds_including_compile": None, "train_seconds": 0.,
                      "input_tokens": 0, "steady_seconds": 0., "steady_tokens": 0,
                      "optimizer_compile": optimizer_compile}
        state["invocations"].append(invocation)
        first_full_shape_seen = False
        stop_reason = None
        torch.cuda.reset_peak_memory_stats()
        with (args.output / "training.jsonl").open("x") as log:
            while state["completed_steps"] < schedule["total_steps"]:
                if args.stop_after is not None and state["completed_steps"] >= args.stop_after:
                    stop_reason = "stop_after"
                    break
                if args.max_seconds is not None and time.perf_counter() - started_wall >= args.max_seconds:
                    stop_reason = "max_seconds"
                    break
                loop_begin = time.perf_counter()
                step = state["completed_steps"]
                start = step * config["effective_batch"]
                indices = state["order"][start:start + config["effective_batch"]]
                rate = learning_rate(schedule, step)
                if config.get("compiled_optimizer", False):
                    set_lr(optimizer, rate)
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                first_full = not first_full_shape_seen and len(indices) >= config["micro_batch"]
                telemetry.phase = "training_initialization" if first_full else "training"
                torch.cuda.synchronize()
                begin = time.perf_counter()
                loss, norm, inputs, targets, applied = update(model, forward, optimizer, train, provider, indices,
                                                             config["micro_batch"], config["mask_policy"],
                                                             step_callable=step_callable)
                torch.cuda.synchronize()
                duration = time.perf_counter() - begin
                if first_full:
                    first_full_shape_seen = True
                    invocation["first_full_shape_update_seconds_including_compile"] = duration
                    check_optimizer_precision(model, optimizer)
                    print(json.dumps({"event": "first_full_shape_update", "seconds_including_compile": duration,
                                      "compile_scope": backend_metadata["compile_scope"]}), flush=True)
                state["completed_steps"] += 1
                state["optimizer_updates"] += int(applied)
                state["trained_tokens"] += inputs
                state["supervised_tokens"] += targets
                invocation["input_tokens"] += inputs
                record = {"step": state["completed_steps"], "loss": float(loss), "grad_norm": float(norm),
                          "lr": rate, "input_tokens": inputs, "supervised_tokens": targets,
                          "cumulative_tokens": state["trained_tokens"], "seconds": duration,
                          "optimizer_applied": applied, "first_full_shape_update": first_full}
                log.write(json.dumps(record) + "\n")
                if state["completed_steps"] % args.log_every == 0 or state["completed_steps"] == schedule["total_steps"]:
                    log.flush()
                    cpt.write(args.output / "progress.json", record)
                    print(json.dumps({"event": "training", **record}), flush=True)
                loop_duration = time.perf_counter() - loop_begin
                state["train_seconds"] += loop_duration
                invocation["train_seconds"] += loop_duration
                if not first_full:
                    state["steady_seconds"] += loop_duration
                    state["steady_tokens"] += inputs
                    invocation["steady_seconds"] += loop_duration
                    invocation["steady_tokens"] += inputs
                if state["completed_steps"] == max(1, schedule["total_steps"] // 2):
                    telemetry.phase = "validation"
                    metrics = dict(step=state["completed_steps"], **evaluate(model, validation, provider))
                    state["validation_history"].append(metrics)
                    cpt.write(args.output / "validation_history.json", state["validation_history"])
                    print(json.dumps({"event": "validation", **metrics}), flush=True)
                if args.checkpoint_every is not None and state["completed_steps"] % args.checkpoint_every == 0 and state["completed_steps"] < schedule["total_steps"]:
                    telemetry.phase = "saving"
                    log.flush()
                    saved = save_checkpoint(args.output / f"checkpoint_step_{state['completed_steps']:08d}",
                                            model, tokenizer, optimizer, state, backend_metadata)
                    cpt.write(args.output / "latest_checkpoint.json", {"path": str(saved), "step": state["completed_steps"]})
                    print(json.dumps({"event": "checkpoint", "path": str(saved), "step": state["completed_steps"]}), flush=True)
        complete = state["completed_steps"] == schedule["total_steps"]
        check_optimizer_precision(model, optimizer)
        actual_inputs, actual_targets = selected_token_counts(train, state["order"], state["completed_steps"], config["effective_batch"])
        if state["trained_tokens"] != actual_inputs or state["supervised_tokens"] != actual_targets:
            raise RuntimeError("Training counters differ from the completed shuffled data prefix")
        if complete and state["trained_tokens"] != expected_inputs:
            raise RuntimeError("Completed training did not consume its configured exact token budget")
        telemetry.phase = "validation_final"
        final_metrics = dict(step=state["completed_steps"], **evaluate(model, validation, provider))
        state["validation_history"].append(final_metrics)
        cpt.write(args.output / "validation_history.json", state["validation_history"])
        print(json.dumps({"event": "validation_final", **final_metrics}), flush=True)
        invocation["end_step"] = state["completed_steps"]
        invocation["stop_reason"] = stop_reason
        telemetry.phase = "saving"
        checkpoint = save_checkpoint(args.output / ("final_checkpoint" if complete else "stopped_checkpoint"),
                                     model, tokenizer, optimizer, state, backend_metadata)
        cpt.write(args.output / "latest_checkpoint.json", {"path": str(checkpoint), "step": state["completed_steps"]})
        result = {"status": "complete" if complete else "stopped", "stop_reason": stop_reason,
                  "completed_steps": state["completed_steps"], "optimizer_updates": state["optimizer_updates"],
                  "configured_steps": schedule["total_steps"], "trained_tokens": state["trained_tokens"],
                  "supervised_tokens": state["supervised_tokens"], "configured_input_tokens": expected_inputs,
                  "dataset_input_tokens": len(train.tokens), "all_dataset_tokens_consumed": complete and state["trained_tokens"] == len(train.tokens),
                  "train_seconds_excluding_validation_and_save": state["train_seconds"],
                  "tokens_per_second_including_first_update": state["trained_tokens"] / state["train_seconds"] if state["train_seconds"] else None,
                  "steady_tokens_per_second": state["steady_tokens"] / state["steady_seconds"] if state["steady_seconds"] else None,
                  "steady_tokens": state["steady_tokens"], "steady_seconds": state["steady_seconds"],
                  "steady_definition": "Exclude the first full-shape update in each invocation; loop timings include data/H2D, synchronization and logging, and exclude validation/save sections",
                  "invocations": state["invocations"], "schedule": schedule, "backend": backend_metadata,
                  "optimizer_compile": optimizer_compile,
                  "validation_before": state["validation_before"], "validation_after": final_metrics,
                  "validation_history": state["validation_history"],
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "gpu_training_this_invocation": telemetry.summary("training"),
                  "checkpoint": str(checkpoint), "full_parameter_training": True,
                  "wall_seconds_this_invocation": time.perf_counter() - started_wall,
                  "checkpoint_verification": "Atomic publication; independent reload/resume verification is separate"}
        cpt.write(args.output / "result.json", result)
        print(json.dumps({"event": "training_complete" if complete else "training_stopped", **result}), flush=True)
    except Exception as error:
        cpt.write(args.output / "result.json", {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        telemetry.stop.set()
        telemetry.thread.join(timeout=6)


if __name__ == "__main__":
    main()
