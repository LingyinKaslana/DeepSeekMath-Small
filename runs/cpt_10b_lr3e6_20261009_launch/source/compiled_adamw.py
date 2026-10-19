"""Compiled foreach AdamW with FP32 parameters, gradients, and moments.

The learning rate is a CUDA scalar tensor updated in place, so a schedule does
not specialize the compiled optimizer to a new Python float at every step.
Compilation is lazy: the returned metadata separates wrapper setup time from
the first step's wall time, which includes compilation and device execution.
"""
import math
import time

import torch


def make_compiled_adamw(parameters, lr=1e-5):
    """Return ``(optimizer, step_callable, mutable_metadata)`` for CUDA FP32."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError("AdamW requires at least one parameter")
    if any(not isinstance(parameter, torch.Tensor) for parameter in parameters):
        raise TypeError("Expected parameter tensors rather than parameter groups")
    device = parameters[0].device
    if device.type != "cuda" or any(parameter.device != device for parameter in parameters):
        raise ValueError("Compiled AdamW requires parameters on one CUDA device")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Compiled AdamW requires FP32 parameters and optimizer moments")
    lr = float(lr)
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("The learning rate must be finite and nonnegative")
    lr_tensor = torch.tensor(lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(parameters, lr=lr_tensor, betas=(0.9, 0.95),
                                  weight_decay=0.1, capturable=True,
                                  foreach=True, fused=False)
    started = time.perf_counter()
    compiled = torch.compile(optimizer.step, fullgraph=False, dynamic=False, mode="default")
    metadata = {
        "optimizer": "torch.optim.AdamW",
        "implementation": "compiled_foreach",
        "compile": {"fullgraph": False, "dynamic": False, "mode": "default"},
        "compile_wrapper_setup_seconds": time.perf_counter() - started,
        "first_step_seconds_including_compilation": None,
        "compile_seconds": None,
        "compile_time_note": "Lazy compilation; first-step time includes device execution and is not pure compilation time",
        "precision": {"parameters": "float32", "gradients": "float32",
                      "exp_avg": "float32", "exp_avg_sq": "float32",
                      "learning_rate": "CUDA float32 scalar"},
        "device": str(device),
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(parameter.numel() for parameter in parameters),
        "capturable": True,
        "foreach": True,
        "fused": False,
        "betas": [0.9, 0.95],
        "weight_decay": 0.1,
        "eps": 1e-8,
    }

    def step():
        if metadata["first_step_seconds_including_compilation"] is None:
            # Synchronize only the first invocation to make lazy-compilation
            # timing reviewable.  Steady-state calls do not add synchronization.
            torch.cuda.synchronize(device)
            first_started = time.perf_counter()
            result = compiled()
            torch.cuda.synchronize(device)
            metadata["first_step_seconds_including_compilation"] = time.perf_counter() - first_started
            return result
        return compiled()

    return optimizer, step, metadata


@torch.no_grad()
def set_lr(optimizer, lr):
    """Fill existing CUDA LR scalars without replacing their tensor identity."""
    lr = float(lr)
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("The learning rate must be finite and nonnegative")
    for group in optimizer.param_groups:
        current = group["lr"]
        if not isinstance(current, torch.Tensor) or current.device.type != "cuda" or current.numel() != 1:
            raise ValueError("Expected an existing one-element CUDA learning-rate tensor")
        current.fill_(lr)


def _comparison(actual, expected, atol, rtol):
    actual = actual.detach()
    expected = expected.detach()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "passed": finite and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "finite": finite,
        "max_absolute_difference": float(difference.max()),
        "relative_l2_difference": float((actual - expected).norm() / expected.norm().clamp_min(1e-30)),
        "atol": atol,
        "rtol": rtol,
        "dtype": str(actual.dtype),
    }


def verify_compiled_adamw(device="cuda"):
    """Compare three updates to ordinary foreach AdamW on small CUDA tensors.

    This check exercises the optimizer math and changing tensor learning rates;
    it does not estimate full-model compile time or steady-state performance.
    It uses a private CPU RNG, leaving the experiment's global RNG untouched.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The capturable compiled AdamW verifier requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device="cpu").manual_seed(1847)
    shapes = [(17, 31), (9,), (), (2, 3, 5)]
    initial = [torch.randn(shape, generator=generator, dtype=torch.float32) * scale
               for shape, scale in zip(shapes, (1.0, 0.001, 0.0, 3.0))]
    reference_parameters = [torch.nn.Parameter(value.to(device)) for value in initial]
    compiled_parameters = [torch.nn.Parameter(value.to(device)) for value in initial]
    reference = torch.optim.AdamW(reference_parameters, lr=1e-5, betas=(0.9, 0.95),
                                 weight_decay=0.1, capturable=False, foreach=True, fused=False)
    optimizer, step, metadata = make_compiled_adamw(compiled_parameters, lr=1e-5)
    records = []
    for update_index, learning_rate in enumerate((1e-5, 3.16e-6, 1e-6), start=1):
        reference.param_groups[0]["lr"] = learning_rate
        set_lr(optimizer, learning_rate)
        gradients = [torch.randn(shape, generator=generator, dtype=torch.float32) * scale
                     for shape, scale in zip(shapes, (0.1, 1e-10, 1e-8, 2.0))]
        if update_index == 1:
            gradients[1].zero_()  # A present zero gradient still applies weight decay.
        for reference_parameter, compiled_parameter, gradient in zip(
                reference_parameters, compiled_parameters, gradients):
            reference_parameter.grad = gradient.to(device)
            compiled_parameter.grad = gradient.to(device)
        reference.step()
        step()
        torch.cuda.synchronize(device)
        comparisons = []
        for index, (reference_parameter, compiled_parameter) in enumerate(zip(
                reference_parameters, compiled_parameters)):
            reference_state = reference.state[reference_parameter]
            compiled_state = optimizer.state[compiled_parameter]
            checks = {
                "parameter": _comparison(compiled_parameter, reference_parameter, atol=2e-7, rtol=0.0),
                "exp_avg": _comparison(compiled_state["exp_avg"], reference_state["exp_avg"], atol=1e-12, rtol=1e-5),
                "exp_avg_sq": _comparison(compiled_state["exp_avg_sq"], reference_state["exp_avg_sq"], atol=1e-15, rtol=1e-5),
                "step": _comparison(compiled_state["step"], reference_state["step"].to(device), atol=0.0, rtol=0.0),
            }
            fp32 = all(compiled_state[name].dtype == torch.float32 for name in ("exp_avg", "exp_avg_sq"))
            comparisons.append({"parameter_index": index, "shape": list(shapes[index]),
                                "checks": checks, "moments_fp32": fp32,
                                "passed": fp32 and all(check["passed"] for check in checks.values())})
        records.append({"update": update_index, "learning_rate": learning_rate,
                        "comparisons": comparisons,
                        "passed": all(comparison["passed"] for comparison in comparisons)})
    return {"status": "passed" if all(record["passed"] for record in records) else "failed",
            "steps": 3, "device": str(device), "records": records, "metadata": metadata}
