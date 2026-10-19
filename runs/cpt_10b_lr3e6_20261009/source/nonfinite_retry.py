"""An optional, single native recomputation before any optimizer mutation.

The caller prepares one immutable set of batches and supplies two backward
closures. A failed attempt never applies the optimizer, advances the data
cursor, or consumes an additional RNG sequence. Only nonfinite loss/norm
failures are retryable; other exceptions propagate immediately.
"""
import math

import torch


class NonfiniteUpdateError(RuntimeError):
    def __init__(self, cause, loss, norm=None, original_error=None):
        self.cause, self.loss, self.norm = cause, loss, norm
        self.original_error = original_error
        super().__init__(f"Nonfinite training {cause}: loss={loss}, gradient_norm={norm}")


def _number(value):
    return None if value is None else float(value.detach())


def _json_number(value):
    # Keep strict JSON while preserving whether the bad value was NaN or Inf.
    if value is None or math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")


def capture_rng(device):
    state = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def nonfinite_gradients(model):
    """Failure-only evidence; never scan all model tensors on a finite update."""
    tensors, elements, present = [], 0, 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        present += 1
        count = int((~torch.isfinite(gradient)).sum())
        if count:
            tensors.append({"name": name, "nonfinite_elements": count,
                            "elements": gradient.numel(), "dtype": str(gradient.dtype)})
            elements += count
    return {"gradient_tensors_present": present, "nonfinite_tensor_count": len(tensors),
            "nonfinite_element_count": elements, "tensors": tensors}


def _backward_and_check(model, optimizer, backward):
    optimizer.zero_grad(set_to_none=True)
    loss = backward()
    # Keep the native strict norm check. In particular, never silently replace
    # error_if_nonfinite=True with a clipping/skipping policy.
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        # Do not reinterpret unrelated runtime failures as numerical failures.
        if "The total norm of order" not in str(error) or "is non-finite" not in str(error):
            raise
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=False)
        cause = "loss_and_gradient_norm" if not bool(torch.isfinite(loss)) else "gradient_norm"
        raise NonfiniteUpdateError(cause, _number(loss), _number(norm), str(error)) from None
    if not bool(torch.isfinite(loss)):
        raise NonfiniteUpdateError("loss", _number(loss), _number(norm))
    return loss.detach(), norm.detach()


def checked_optimizer_update(model, optimizer, backward, *, native_backward=None,
                             retry_native=False, step_callable=None, step=None,
                             event_callback=None):
    """Return finite loss/norm after exactly one optimizer update, or raise.

RNG is captured only when retry is enabled. Parameters/optimizer states are
never touched by a failed backward/check attempt. ``native_backward`` must
temporarily restore the caller's native backend and its precision policy.
"""
    device = next(model.parameters()).device
    rng = capture_rng(device) if retry_native else None
    retry_event = None
    try:
        loss, norm = _backward_and_check(model, optimizer, backward)
    except NonfiniteUpdateError as error:
        if not retry_native:
            raise
        if native_backward is None:
            raise ValueError("Native retry requires a native backward closure") from error
        retry_event = {"event": "native_retry", "step": step, "cause": error.cause,
                       "fast_loss": _json_number(error.loss), "fast_grad_norm": _json_number(error.norm),
                       "fast_error": error.original_error or str(error),
                       "fast_nonfinite_gradients": nonfinite_gradients(model)}
    if retry_event is not None:
        restore_rng(rng, device)
        try:
            loss, norm = _backward_and_check(model, optimizer, native_backward)
        except Exception as error:
            native_gradient_evidence = nonfinite_gradients(model)
            optimizer.zero_grad(set_to_none=True)
            restore_rng(rng, device)
            retry_event.update(status="failed_before_optimizer", optimizer_applied=False,
                               native_loss=_json_number(getattr(error, "loss", None)),
                               native_grad_norm=_json_number(getattr(error, "norm", None)),
                               native_error=getattr(error, "original_error", None) or str(error),
                               native_nonfinite_gradients=native_gradient_evidence)
            if event_callback is not None:
                event_callback(retry_event)
            raise
        retry_event.update(status="finite_before_optimizer", optimizer_applied=False,
                           native_loss=_number(loss), native_grad_norm=_number(norm))
        if event_callback is not None:
            event_callback(retry_event)
    if step_callable is None:
        optimizer.step()
    else:
        step_callable()
    return loss, norm
