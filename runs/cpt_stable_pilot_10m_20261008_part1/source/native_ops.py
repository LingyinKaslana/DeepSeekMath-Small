"""Opaque native SiLU operators for numerically faithful compiled training.

Inductor can decompose SiLU's backward into several BF16 primitives. PyTorch's
CUDA kernel computes that derivative in FP32 and rounds once on output. These
operators retain the native forward/backward kernels inside a compiled graph.
Only first-order autograd is needed/supported by this CPT training wrapper.
"""
import torch
from torch import nn
from torch.nn import functional as F


@torch.library.custom_op("deepseekmath_native::silu", mutates_args=())
def native_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x, inplace=False)


@native_silu.register_fake
def _native_silu_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("deepseekmath_native::silu_backward", mutates_args=())
def native_silu_backward(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward.default(grad_output, x)


@native_silu_backward.register_fake
def _native_silu_backward_fake(grad_output, x):
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return native_silu_backward(grad_output, x)


native_silu.register_autograd(_backward, setup_context=_setup_context)


class NativeSiLU(nn.Module):
    """Parameter-free, instance-local replacement for HF's SiLU activation."""

    def forward(self, x):
        return native_silu(x)
