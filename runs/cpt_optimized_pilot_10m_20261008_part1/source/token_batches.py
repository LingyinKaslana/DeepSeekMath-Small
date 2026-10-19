"""Vectorized, bounded host-to-device batches for ``PackedTokens``.

This provider keeps only requested token blocks on the GPU.  Each call gets
fresh pinned staging rather than reusing a buffer while an asynchronous copy
may still be reading it.  PyTorch's pinned allocator records transfer events
and defers recycling that allocation until the copy is complete.
"""
from collections import OrderedDict

import numpy as np
import torch


class PackedTokenBatches:
    """Callable replacement for ``cpt.batch(dataset, indices)``.

    ``device='cpu'`` also supports exact data-path checks without a GPU.  The
    CPU mode uses ordinary host allocations; CUDA uses pinned int64 staging.
    Input IDs and labels alias for complete blocks, as both are read-only
    during causal-LM forward/backward.  Callers must not mutate either tensor.
    """

    def __init__(self, device="cuda", max_cached_masks=8):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if max_cached_masks < 1:
            raise ValueError("At least one attention-mask cache entry is required")
        self.max_cached_masks = int(max_cached_masks)
        self._masks = OrderedDict()
        self._positions = {}

    def _allocate_host(self, shape):
        return torch.empty(shape, dtype=torch.int64,
                           pin_memory=self.device.type == "cuda")

    def _attention_mask(self, shape):
        key = tuple(shape)
        mask = self._masks.pop(key, None)
        if mask is None:
            mask = torch.ones(key, dtype=torch.int64, device=self.device)
        self._masks[key] = mask
        while len(self._masks) > self.max_cached_masks:
            self._masks.popitem(last=False)
        return mask

    def clear_cache(self):
        """Release cached masks and CPU position arrays between experiments."""
        self._masks.clear()
        self._positions.clear()

    def __call__(self, dataset, indices):
        index_array = np.asarray(indices)
        if index_array.ndim != 1 or index_array.size == 0:
            raise ValueError("Batch indices must be a nonempty one-dimensional collection")
        if not np.issubdtype(index_array.dtype, np.integer):
            raise TypeError("Batch indices must be integers")
        if np.any(index_array < 0) or np.any(index_array >= len(dataset)):
            raise IndexError("Batch indices contain an out-of-range block")
        index_array = index_array.astype(np.int64, copy=False)
        sequence_length = int(dataset.sequence_length)
        token_count = len(dataset.tokens)
        shape = (len(index_array), sequence_length)
        # All counts are derived before transfer.  In particular, computing
        # them does not launch a GPU reduction or synchronize the CUDA stream.
        starts = index_array * sequence_length
        lengths = np.minimum(sequence_length, token_count - starts)
        input_tokens = int(lengths.sum())
        valid_targets = int(np.maximum(lengths - 1, 0).sum())
        positions = self._positions.get(sequence_length)
        if positions is None:
            positions = np.arange(sequence_length, dtype=np.int64)
            # Bound this smaller CPU cache as well when one provider is reused
            # across a sequence-length sweep.
            if len(self._positions) >= self.max_cached_masks:
                self._positions.pop(next(iter(self._positions)))
            self._positions[sequence_length] = positions
        offsets = starts[:, None] + positions[None, :]
        partial = lengths != sequence_length
        if partial.any():
            if dataset.pad_token_id is None:
                raise ValueError("A partial token block requires pad_token_id")
            # Clip only the synthetic trailing offsets.  Every real offset
            # remains unchanged, including in duplicate or shuffled blocks.
            np.minimum(offsets, token_count - 1, out=offsets)
        host_ids = self._allocate_host(shape)
        # np.take casts uint32 memmap values directly into pinned int64 output,
        # avoiding individual Dataset rows, row clones, and torch.stack.
        # Bounds were checked above; clip mode avoids np.take's extra output
        # buffering for exception safety in its default "raise" mode.
        np.take(dataset.tokens, offsets, out=host_ids.numpy(), mode="clip")
        if partial.any():
            padding = positions[None, :] >= lengths[:, None]
            host_ids.numpy()[padding] = dataset.pad_token_id
            host_labels = self._allocate_host(shape)
            host_labels.copy_(host_ids)
            host_labels.numpy()[padding] = -100
        else:
            host_labels = host_ids
        input_ids = host_ids.to(self.device, non_blocking=True)
        labels = input_ids if host_labels is host_ids else host_labels.to(self.device, non_blocking=True)
        values = {"input_ids": input_ids, "labels": labels}
        if getattr(dataset, "explicit_mask", False):
            # Match the existing all-ones mask, including trailing padding.
            # Causality protects real tokens and ignored labels remove padding
            # from the loss; the 2D mask keeps HF on native FlashAttention.
            values["attention_mask"] = self._attention_mask(shape)
        return values, valid_targets, input_tokens
