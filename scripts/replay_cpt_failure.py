"""Isolated CPT failure diagnostics using an immutable training source snapshot.

Quick mode compares the failed batch at the latest saved weights without an
optimizer update. Replay mode restores the original AdamW/order/RNG and stops
before applying any nonfinite-gradient update. It never edits the source run.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def emit(value):
    print(json.dumps(clean(value), allow_nan=False), flush=True)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gradient_summary(model, torch):
    """Inspect unchanged gradients, including a norm that avoids FP32 squaring."""
    rows = []
    squares64 = torch.zeros((), dtype=torch.float64, device="cuda")
    norms32 = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            rows.append({"name": name, "gradient": None})
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        all_finite = bool(finite.all())
        norm32 = torch.linalg.vector_norm(gradient)
        norm64 = torch.linalg.vector_norm(gradient, dtype=torch.float64)
        norms32.append(norm32)
        squares64 += norm64.square()
        row = {"name": name, "dtype": str(gradient.dtype), "numel": gradient.numel(),
               "all_finite": all_finite, "max_abs": float(gradient.abs().max()),
               "norm_fp32": float(norm32), "norm_fp64": float(norm64)}
        if not all_finite:
            row.update(nonfinite_elements=int((~finite).sum()),
                       nan_elements=int(torch.isnan(gradient).sum()),
                       inf_elements=int(torch.isinf(gradient).sum()))
        rows.append(row)
    total32 = torch.linalg.vector_norm(torch.stack(norms32)) if norms32 else torch.zeros((), device="cuda")
    total64 = squares64.sqrt()
    return {"all_gradients_finite": all(row.get("all_finite", True) for row in rows),
            "tensor_count": len(norms32), "total_norm_fp32": float(total32),
            "total_norm_fp64": float(total64),
            "nonfinite_tensor_names": [row["name"] for row in rows if row.get("all_finite") is False],
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mode", choices=("quick", "replay"), default="quick")
    parser.add_argument("--target-step", type=int, default=175997, help="One-based global update")
    parser.add_argument("--end-step", type=int, default=176020)
    parser.add_argument("--include-strict", action="store_true")
    parser.add_argument("--save-pre-failure-model", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = json.loads((args.run / "backend_config.json").read_text())
    saved_metadata = json.loads((args.checkpoint / "checkpoint_manifest.json").read_text())
    source = args.source or args.run / "source"
    source_hashes = {name: sha(source / name) for name in manifest["source_sha256"]}
    if source_hashes != manifest["source_sha256"]:
        raise RuntimeError("Diagnostic source snapshot differs from the original run")
    if saved_metadata["signature"] != manifest["signature"]:
        raise RuntimeError("Checkpoint signature differs from original run")
    if sha(args.run / "backend_config.json") != manifest["signature"]["backend_config_sha256"]:
        raise RuntimeError("Saved backend configuration checksum changed")
    sys.path.insert(0, str(source.resolve()))
    import torch
    import train_cpt as trainer
    from transformers import AutoTokenizer
    from cpt_orders import EpochOrders, load_saved_orders, order_indices
    from compiled_adamw import make_compiled_adamw, set_lr
    from model_backends import configure_model
    from packed_data import PackedTokens
    from token_batches import PackedTokenBatches

    torch.set_num_threads(8)
    torch.manual_seed(manifest["arguments"]["seed"])
    provenance = {"mode": args.mode, "run": str(args.run.resolve()),
                  "checkpoint": str(args.checkpoint.resolve()),
                  "checkpoint_step": saved_metadata["completed_steps"], "target_step": args.target_step,
                  "source_directory": str(source.resolve()), "source_sha256": source_hashes,
                  "diagnostic_script_sha256": sha(Path(__file__)),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "precision": "BF16 forward/autocast; backward outside autocast; FP32 parameters and gradients",
                  "flash_deterministic": torch.are_deterministic_algorithms_enabled(),
                  "limitation": "Production Flash attention is nondeterministic; restored replay need not be bitwise-identical.",
                  "full_token_file_checksum_revalidation": False,
                  "production_files_modified": False}
    write(args.output / "provenance.json", provenance)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    data_path = Path(manifest["arguments"]["data"])
    train = PackedTokens(data_path, sequence_length=config["sequence_length"], include_tail=True,
                         pad_token_id=tokenizer.eos_token_id)
    train.explicit_mask = config["mask_policy"] == "ones"
    provider = PackedTokenBatches()
    orders = EpochOrders(args.checkpoint / "orders", saved_metadata["order_metadata"])
    target_indices = order_indices(orders, args.target_step - 1, config["effective_batch"])
    write(args.output / "target_batch.json", {"global_step": args.target_step,
          "epoch": (args.target_step - 1) // orders.epoch_steps + 1,
          "step_in_epoch": (args.target_step - 1) % orders.epoch_steps + 1,
          "indices": [int(i) for i in target_indices],
          "tail_in_batch": bool((target_indices == len(train) - 1).any())})
    model = trainer.cpt.load_model(argparse.Namespace(model=str(args.checkpoint), revision=None))
    trainer.check_model_safety(model)
    model.train()

    def compare(indices, prefix):
        cases = [("fast", config["options"]),
                 ("native", dict(config["options"], compile_scope="none", preserve_silu=False))]
        if args.include_strict:
            cases.append(("strict_mlp", dict(config["options"], compile_scope="mlp", preserve_silu=True)))
        result = []
        for name, options in cases:
            model.zero_grad(set_to_none=True)
            forward, metadata = configure_model(model, options)
            values, targets, inputs = provider(train, indices)
            if config["mask_policy"] == "causal":
                values["attention_mask"] = {"full_attention": None}
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = forward(**values, use_cache=False).loss
            loss.backward()
            torch.cuda.synchronize()
            summary = gradient_summary(model, torch)
            record = {"case": name, "weights_updated": False, "loss": float(loss.detach()),
                      "loss_finite": bool(torch.isfinite(loss.detach())),
                      "seconds_including_compile_and_grad_audit": time.perf_counter() - started,
                      "input_tokens": inputs, "supervised_tokens": targets,
                      "backend": metadata, **summary}
            write(args.output / f"{prefix}_{name}.json", record)
            emit({key: value for key, value in record.items() if key not in ("rows", "backend", "nonfinite_tensor_names")})
            result.append({key: value for key, value in record.items() if key not in ("rows", "backend")})
            del loss, values, forward
            gc.collect()
        model.zero_grad(set_to_none=True)
        return result

    if args.mode == "quick":
        result = compare(target_indices, "quick")
        write(args.output / "result.json", {"status": "complete", "mode": "quick", "cases": result,
                                             "weights_step": saved_metadata["completed_steps"]})
        return

    state = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    orders = load_saved_orders(state, args.checkpoint)
    trainer.validate_resume(state, manifest["signature"], train,
        argparse.Namespace(epochs=manifest["epochs"], seed=manifest["arguments"]["seed"],
                           steps=None, lr=manifest["arguments"]["lr"]), orders=orders)
    forward, metadata = configure_model(model, config["options"])
    optimizer, step_callable, _ = make_compiled_adamw(model.parameters(), manifest["arguments"]["lr"])
    trainer.restore_optimizer(optimizer, state["optimizer"])
    trainer.check_optimizer_precision(model, optimizer)
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    start_step = state["completed_steps"]
    schedule = state["schedule"]
    del state
    gc.collect()
    original_clip = torch.nn.utils.clip_grad_norm_
    context = {}
    failed = None

    def observed_forward(*positional, **keyword):
        output = forward(*positional, **keyword)
        context["loss"] = output.loss.detach()
        return output

    def observed_clip(parameters, *positional, **keyword):
        parameters = list(parameters)
        if context["step"] == args.target_step:
            summary = gradient_summary(model, torch)
            summary.update(step=context["step"], loss=float(context["loss"]),
                           loss_finite=bool(torch.isfinite(context["loss"])))
            write(args.output / "target_gradients_before_clip.json", summary)
            emit({"event": "target_gradient_audit", **{key: value for key, value in summary.items()
                                                      if key not in ("rows", "nonfinite_tensor_names")}})
        try:
            return original_clip(parameters, *positional, **keyword)
        except RuntimeError:
            context["clip_failed"] = True
            summary = gradient_summary(model, torch)
            summary.update(step=context["step"], loss=float(context["loss"]),
                           loss_finite=bool(torch.isfinite(context["loss"])))
            write(args.output / "failed_gradients.json", summary)
            emit({key: value for key, value in summary.items() if key not in ("rows", "nonfinite_tensor_names")})
            raise

    torch.nn.utils.clip_grad_norm_ = observed_clip
    try:
        with (args.output / "replay_training.jsonl").open("x") as stream:
            for zero_step in range(start_step, min(args.end_step, schedule["total_steps"])):
                context["step"] = zero_step + 1
                context["clip_failed"] = False
                indices = order_indices(orders, zero_step, config["effective_batch"])
                set_lr(optimizer, trainer.learning_rate(schedule, zero_step))
                begin = time.perf_counter()
                try:
                    loss, norm, inputs, targets, applied = trainer.update(model, observed_forward, optimizer,
                        train, provider, indices, config["micro_batch"], config["mask_policy"],
                        step_callable=step_callable)
                    torch.cuda.synchronize()
                except Exception as error:
                    failed = {"step": zero_step + 1, "last_successful_step": zero_step,
                              "error": str(error), "traceback": traceback.format_exc(),
                              "bad_gradient_optimizer_update_applied": False if context["clip_failed"] else "unknown",
                              "indices": [int(index) for index in indices]}
                    write(args.output / "failure.json", failed)
                    if args.save_pre_failure_model and context["clip_failed"]:
                        model.save_pretrained(args.output / "pre_failure_model", safe_serialization=True)
                        tokenizer.save_pretrained(args.output / "pre_failure_model")
                    # No update occurred after the failing clip; every case below
                    # sees the same last-successful-step weights and data batch.
                    failed["same_weight_comparisons"] = compare(indices, "failed_step")
                    break
                record = {"step": zero_step + 1, "loss": float(loss), "grad_norm": float(norm),
                          "input_tokens": inputs, "targets": targets, "optimizer_applied": applied,
                          "seconds": time.perf_counter() - begin}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                if (zero_step + 1) % 100 == 0 or zero_step + 1 == args.target_step:
                    emit({"event": "replay", **record})
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    write(args.output / "result.json", {"status": "failure_reproduced" if failed else "no_failure_in_replayed_window",
          "start_step": start_step, "end_step": context.get("step"), "target_step": args.target_step,
          "failure": failed, "production_run_resumed": False,
          "limitation": "Production Flash attention nondeterminism prevents asserting an identical weight trajectory."})


if __name__ == "__main__":
    main()
