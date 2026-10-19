"""Deliberately inject one bad gradient to verify native retry on the real batch.

This is an isolated diagnostic. It never writes the source checkpoint or runs
as a production trainer. The fault is a documented, one-shot parameter hook.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

import cpt
import train_cpt as trainer
from cpt_orders import load_saved_orders, order_indices
from compiled_adamw import make_compiled_adamw, set_lr
from model_backends import configure_model
from packed_data import PackedTokens
from token_batches import PackedTokenBatches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--backend-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(8)
    config = trainer.load_backend(args.backend_config)
    saved = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    orders = load_saved_orders(saved, args.checkpoint)
    step = saved["completed_steps"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    train = PackedTokens(args.data, sequence_length=config["sequence_length"], include_tail=True,
                         pad_token_id=tokenizer.eos_token_id)
    train.explicit_mask = config["mask_policy"] == "ones"
    indices = order_indices(orders, step, config["effective_batch"])
    provider = PackedTokenBatches()
    model = cpt.load_model(argparse.Namespace(model=str(args.checkpoint), revision=None))
    trainer.check_model_safety(model)
    forward, backend = configure_model(model, config["options"])
    optimizer, compiled_step, optimizer_compile = make_compiled_adamw(model.parameters(), 1e-5)
    trainer.restore_optimizer(optimizer, saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    rate = trainer.learning_rate(saved["schedule"], step)
    del saved
    gc.collect()
    set_lr(optimizer, rate)
    counters = {"hook_calls": 0, "injections": 0, "optimizer_calls": 0}
    probe = model.model.norm.weight
    old_probe = probe.detach().clone()

    def inject_first_gradient(gradient):
        counters["hook_calls"] += 1
        if counters["injections"] == 0:
            counters["injections"] += 1
            gradient = gradient.clone()
            gradient.view(-1)[0] = float("nan")
        return gradient

    def counted_step():
        counters["optimizer_calls"] += 1
        compiled_step()

    events = []
    hook = probe.register_hook(inject_first_gradient)
    try:
        loss, norm, inputs, targets, applied = trainer.update(
            model, forward, optimizer, train, provider, indices, config["micro_batch"],
            config["mask_policy"], step_callable=counted_step,
            retry_native_on_nonfinite=True, step=step + 1, retry_event_callback=events.append)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    steps = {int(state["step"].item()) for state in optimizer.state.values()}
    assert counters == {"hook_calls": 2, "injections": 1, "optimizer_calls": 1}, counters
    assert len(events) == 1 and events[0]["status"] == "finite_before_optimizer", events
    assert events[0]["cause"] == "gradient_norm", events
    assert inputs == 32768 and targets == 32752 and applied
    assert steps == {step + 1}, steps
    assert torch.isfinite(loss) and torch.isfinite(norm)
    assert not torch.equal(old_probe, probe)
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())
    trainer.check_optimizer_precision(model, optimizer)
    result = {"status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint.resolve()), "global_step": step + 1,
              "shape": [config["micro_batch"], config["sequence_length"]],
              "batch_indices": [int(i) for i in indices], "input_tokens": inputs,
              "supervised_tokens": targets, "loss": float(loss), "grad_norm": float(norm),
              "counters": counters, "optimizer_state_steps": sorted(steps),
              "events": events, "precision": "BF16 autocast; FP32 master/gradient/AdamW",
              "source_sha256": {name: cpt.sha(Path(trainer.__file__).parent / name)
                                 for name in trainer.SOURCE_FILES},
              "diagnostic_source_sha256": cpt.sha(Path(__file__)),
              "backend_config_sha256": cpt.sha(args.backend_config),
              "backend": backend, "optimizer_compile": optimizer_compile,
              "seconds": time.perf_counter() - started,
              "scope": "Documented NaN injection in first backward only; native same-batch retry, one real compiled AdamW update; no production files modified"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "global_step", "counters", "loss", "grad_norm", "seconds")}))


if __name__ == "__main__":
    main()
