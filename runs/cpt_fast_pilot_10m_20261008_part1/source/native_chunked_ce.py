"""Native PyTorch CE with bounded FP32 intermediates and recomputed backward.

The input logits stay in their original dtype. Forward saves only those logits,
targets, and one FP32 normalization scalar; backward recreates native FP32
cross-entropy for each row chunk and writes gradients in the logits dtype.
This retains the complete low-precision logits tensor but avoids retaining
complete FP32 logits and log-softmax tensors. It trades recomputation and extra
kernel launches for memory; throughput benefits require a separate measurement.

The chunked scalar loss reduction may round differently from a full native mean.
The backward seed is upstream / total valid-target count, matching native CE
mean semantics. Numerical verification is still required on the actual device.
Only first derivatives are supported. An all-ignored/empty mean returns NaN,
as native PyTorch CE does; callers should skip zero-target training updates.
"""

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


class NativeChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, chunk_rows, ignore_index, normalizer):
        if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
            raise ValueError('Expected [rows, vocabulary] logits and [rows] targets')
        if logits.shape[1] < 1:
            raise ValueError('The logits vocabulary must be nonempty')
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Supported logits dtypes are float16, bfloat16, and float32')
        if targets.dtype != torch.long or targets.device != logits.device:
            raise ValueError('Targets must be torch.long on the logits device')
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ValueError('chunk_rows must be a positive integer')
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise ValueError('ignore_index must be an integer')
        with torch.autocast(logits.device.type, enabled=False):
            if normalizer is None:
                divisor = (targets != ignore_index).sum().to(dtype=torch.float32)
            else:
                divisor = torch.as_tensor(normalizer, device=logits.device, dtype=torch.float32)
                if divisor.numel() != 1:
                    raise ValueError('The optional normalizer must be scalar')
                divisor = divisor.reshape(())
            loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
            for start in range(0, len(logits), chunk_rows):
                loss_sum.add_(F.cross_entropy(logits[start:start + chunk_rows].float(),
                                             targets[start:start + chunk_rows],
                                             ignore_index=ignore_index, reduction='sum'))
            loss = loss_sum / divisor
        ctx.save_for_backward(logits, targets, divisor)
        ctx.chunk_rows, ctx.ignore_index = chunk_rows, ignore_index
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        logits, targets, divisor = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        with torch.enable_grad(), torch.autocast(logits.device.type, enabled=False):
            # Native NLL mean uses -upstream / total_weight for the target row.
            # Native NLL sum receives this scalar as its upstream seed, so the
            # per-row log-softmax backward retains native arithmetic.
            seed = grad_output.to(dtype=torch.float32) / divisor
            for start in range(0, len(logits), ctx.chunk_rows):
                stop = min(start + ctx.chunk_rows, len(logits))
                chunk = logits[start:stop].detach().float().requires_grad_(True)
                chunk_loss = F.cross_entropy(chunk, targets[start:stop],
                                             ignore_index=ctx.ignore_index, reduction='sum')
                gradient, = torch.autograd.grad(chunk_loss, chunk, grad_outputs=seed,
                                                create_graph=False, retain_graph=False)
                grad_logits[start:stop].copy_(gradient.to(dtype=logits.dtype))
                # Release the last chunk's log-softmax/FP32 gradient before the
                # next chunk allocates its intermediates.
                del chunk, chunk_loss, gradient
        return grad_logits, None, None, None, None


def native_chunked_cross_entropy(logits, targets, chunk_rows=4096, ignore_index=-100, normalizer=None):
    """FP32 CE mean over row logits, or native sum divided by an explicit scalar."""
    return NativeChunkedCE.apply(logits, targets, chunk_rows, ignore_index, normalizer)


def causal_lm_loss(logits, labels, vocab_size=None, num_items_in_batch=None,
                   ignore_index=-100, shift_labels=None, chunk_rows=4096, **kwargs):
    """HF causal-LM loss signature, including shifted labels and global counts.

    Bind this callable as ``model.loss_function`` without replacing the model's
    forward. It expects ordinary [..., vocabulary] logits and unshifted labels,
    or the already-shifted ``shift_labels`` accepted by Transformers.
    """
    vocabulary = logits.shape[-1] if vocab_size is None else vocab_size
    if vocabulary != logits.shape[-1]:
        raise ValueError('vocab_size differs from the logits vocabulary')
    if shift_labels is None:
        if labels is None:
            raise ValueError('labels or shift_labels must be supplied')
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    targets = shift_labels.reshape(-1).to(device=logits.device)
    return native_chunked_cross_entropy(logits.reshape(-1, vocabulary), targets,
                                        chunk_rows=chunk_rows, ignore_index=ignore_index,
                                        normalizer=num_items_in_batch)


apply_loss = causal_lm_loss
